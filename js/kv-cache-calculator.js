/* KV Cache Calculator
 * --------------------
 * Pure-vanilla JS, no dependencies. All math runs in the browser.
 *
 * Two ways to populate the architecture fields:
 *   1. Pick a built-in preset from the "Model preset" dropdown.
 *   2. Type a Hugging Face repo id (e.g. `Qwen/Qwen2.5-7B-Instruct`) and click
 *      Load. We fetch only the few-KB `config.json` and the `/api/models/`
 *      metadata — never weights — and map the relevant fields onto the form.
 *
 * Note on transformers.js: we deliberately do NOT use it. It bundles ONNX
 * Runtime (tens of MB) and is built to *run* models. We only need to read
 * 2-6 KB of metadata, which is a single fetch().
 *
 * "Manual overrides" semantics:
 *   - Any field the user explicitly edits is added to a `manualOverrides` Set.
 *   - When a new preset (built-in or HF) is loaded, fields that are in
 *     `manualOverrides` are left alone; the rest are stamped with the new
 *     baseline values.
 *   - A "↺ reset" chip is rendered next to each overridden field, and a
 *     "↺ reset all overrides" link appears next to the Advanced Options
 *     summary when at least one override is present.
 *
 * Sizing math:
 *   weights  = params * bytes_per_param
 *   kv/token = depends on attention type
 *     MHA/GQA/MQA: 2 * layers * kv_heads * head_dim * bytes_per_kv
 *     MLA        : layers * (kv_lora_rank + qk_rope_head_dim) * bytes_per_kv
 *     SWA mix    : 2 * kv_heads * head_dim * bytes_per_kv * effective_tokens
 *                  per cycle of (local + global) layers
 *     Hybrid     : 2 * attn_layers * kv_heads * head_dim * bytes_per_kv
 *                  (only attention layers contribute)
 *   kv_total = kv_bytes(effective) / paged_efficiency
 *   overhead = weights * (overhead_pct/100) + fixed_overhead_GiB
 *   total    = weights + kv_total + overhead
 */

(function () {
    "use strict";

    // ============================================================
    // Built-in model presets (popular configs, 2026 era)
    // `params` is total parameter count in billions. For MoE models that's
    // total params, since hosting requires loading every expert.
    // ============================================================
    const MODELS = [
        {
            id: "custom",
            label: "Custom configuration",
            attn: "gqa",
            params: 70, layers: 80, qHeads: 64, kvHeads: 8, headDim: 128,
            note: "Tweak any field below to model your own config."
        },

        // --- Llama family (GQA) ---
        {
            id: "llama-3.1-8b",
            label: "Llama 3.1 8B Instruct (GQA)",
            attn: "gqa",
            params: 8.03, layers: 32, qHeads: 32, kvHeads: 8, headDim: 128,
            note: "Meta Llama 3.1 8B. Standard GQA. Context up to 128K."
        },
        {
            id: "llama-3.1-70b",
            label: "Llama 3.1 70B Instruct (GQA)",
            attn: "gqa",
            params: 70, layers: 80, qHeads: 64, kvHeads: 8, headDim: 128,
            note: "Meta Llama 3.1 70B Instruct. GQA-8."
        },
        {
            id: "llama-3.3-70b",
            label: "Llama 3.3 70B Instruct (GQA)",
            attn: "gqa",
            params: 70.55, layers: 80, qHeads: 64, kvHeads: 8, headDim: 128,
            note: "Meta Llama 3.3 70B Instruct. Same arch as 3.1 70B."
        },
        {
            id: "llama-3.1-405b",
            label: "Llama 3.1 405B (GQA)",
            attn: "gqa",
            params: 405, layers: 126, qHeads: 128, kvHeads: 8, headDim: 128,
            note: "Meta Llama 3.1 405B. GQA-8. Multi-node TP usually required."
        },
        {
            id: "llama-4-scout",
            label: "Llama 4 Scout 17B-16E (GQA, MoE)",
            attn: "gqa",
            params: 108.64, layers: 48, qHeads: 40, kvHeads: 8, headDim: 128,
            note: "Llama 4 Scout: 17B active / 16-expert MoE, ~109B total. " +
                  "Uses chunked attention internally; treated as GQA for sizing."
        },
        {
            id: "llama-4-maverick",
            label: "Llama 4 Maverick 17B-128E (GQA, MoE)",
            attn: "gqa",
            params: 401.58, layers: 48, qHeads: 40, kvHeads: 8, headDim: 128,
            note: "Llama 4 Maverick: 17B active / 128-expert MoE, ~402B total."
        },

        // --- Mistral / Mixtral ---
        {
            id: "mistral-7b",
            label: "Mistral 7B v0.3 (GQA)",
            attn: "gqa",
            params: 7.25, layers: 32, qHeads: 32, kvHeads: 8, headDim: 128,
            note: "Mistral 7B v0.3 (sliding window dropped after v0.2)."
        },
        {
            id: "mistral-large-2",
            label: "Mistral Large 2 123B (GQA)",
            attn: "gqa",
            params: 123, layers: 88, qHeads: 96, kvHeads: 8, headDim: 128,
            note: "Mistral Large 2 (Mistral-Large-Instruct-2407)."
        },
        {
            id: "mixtral-8x22b",
            label: "Mixtral 8x22B (GQA, MoE)",
            attn: "gqa",
            params: 141, layers: 56, qHeads: 48, kvHeads: 8, headDim: 128,
            note: "MoE: 8 experts × 22B, ~39B active. Weights size = total params."
        },

        // --- Qwen ---
        {
            id: "qwen2.5-7b",
            label: "Qwen2.5 7B Instruct (GQA)",
            attn: "gqa",
            params: 7.62, layers: 28, qHeads: 28, kvHeads: 4, headDim: 128,
            note: "Qwen2.5 7B Instruct."
        },
        {
            id: "qwen2.5-72b",
            label: "Qwen2.5 72B Instruct (GQA)",
            attn: "gqa",
            params: 72.7, layers: 80, qHeads: 64, kvHeads: 8, headDim: 128,
            note: "Qwen2.5 72B Instruct."
        },
        {
            id: "qwen3-32b",
            label: "Qwen3 32B (GQA)",
            attn: "gqa",
            params: 32, layers: 64, qHeads: 64, kvHeads: 8, headDim: 128,
            note: "Qwen3 32B. Hidden size 5120, GQA-8."
        },
        {
            id: "qwen3-235b-a22b",
            label: "Qwen3 235B-A22B (GQA, MoE)",
            attn: "gqa",
            params: 235.09, layers: 94, qHeads: 64, kvHeads: 4, headDim: 128,
            note: "Qwen3 235B-A22B: ~22B active / 128-expert MoE, ~235B total."
        },

        // --- DeepSeek (MLA) ---
        {
            id: "deepseek-v3",
            label: "DeepSeek-V3 671B (MLA, MoE)",
            attn: "mla",
            params: 671, layers: 61, qHeads: 128, kvHeads: 128, headDim: 128,
            mlaLora: 512, mlaRope: 64,
            note: "DeepSeek-V3. MLA-compressed KV cache; ~37B active / 671B total."
        },
        {
            id: "deepseek-r1",
            label: "DeepSeek-R1 671B (MLA, MoE)",
            attn: "mla",
            params: 671, layers: 61, qHeads: 128, kvHeads: 128, headDim: 128,
            mlaLora: 512, mlaRope: 64,
            note: "DeepSeek-R1 (reasoning). Same architecture as V3."
        },

        // --- Gemma 3 (SWA local/global 5:1) ---
        {
            id: "gemma-3-12b",
            label: "Gemma 3 12B IT (SWA 5:1, 1K window)",
            attn: "swa",
            params: 12.19, layers: 48, qHeads: 16, kvHeads: 8, headDim: 256,
            swaWindow: 1024, swaLocal: 5, swaGlobal: 1,
            note: "Google Gemma 3 12B IT. 5 local-attention layers + 1 global per cycle."
        },
        {
            id: "gemma-3-27b",
            label: "Gemma 3 27B IT (SWA 5:1, 1K window)",
            attn: "swa",
            params: 27.43, layers: 62, qHeads: 32, kvHeads: 16, headDim: 128,
            swaWindow: 1024, swaLocal: 5, swaGlobal: 1,
            note: "Google Gemma 3 27B IT. 5:1 local/global pattern."
        },

        // --- Falcon 40B ---
        {
            id: "falcon-40b",
            label: "Falcon 40B (GQA-8)",
            attn: "gqa",
            params: 41.84, layers: 60, qHeads: 128, kvHeads: 8, headDim: 64,
            note: "TII Falcon 40B (new decoder architecture). GQA with 8 KV heads."
        },

        // --- Hybrid (linear-state + attention) ---
        {
            id: "qwen3-next-80b",
            label: "Qwen3-Next 80B-A3B (Hybrid 3:1)",
            attn: "hybrid",
            params: 80, layers: 48, qHeads: 16, kvHeads: 2, headDim: 256,
            hybridAttn: 12,
            note: "Qwen3-Next: 12 Gated Attention + 36 Gated DeltaNet (3:1). " +
                  "80B total / 3B active."
        },
        {
            id: "jamba-1.5-mini",
            label: "Jamba 1.5 Mini 52B (Hybrid Mamba+Attn)",
            attn: "hybrid",
            params: 52, layers: 32, qHeads: 32, kvHeads: 8, headDim: 128,
            hybridAttn: 4,
            note: "AI21 Jamba 1.5 Mini. ~1:7 attention-to-Mamba ratio. " +
                  "52B total / 12B active."
        }
    ];

    // ============================================================
    // GPU presets (BF16 host VRAM, GiB, current and recent generations)
    // ============================================================
    const GPUS = [
        { id: "h100-80",     label: "NVIDIA H100 80GB (SXM/PCIe)",    gib: 80  },
        { id: "h100-94",     label: "NVIDIA H100 NVL 94GB",           gib: 94  },
        { id: "h200-141",    label: "NVIDIA H200 141GB",              gib: 141 },
        { id: "b200-192",    label: "NVIDIA B200 192GB",              gib: 192 },
        { id: "gb200-192",   label: "NVIDIA GB200 (per Blackwell die) 192GB", gib: 192 },
        { id: "a100-80",     label: "NVIDIA A100 80GB",               gib: 80  },
        { id: "a100-40",     label: "NVIDIA A100 40GB",               gib: 40  },
        { id: "l40s-48",     label: "NVIDIA L40S 48GB",               gib: 48  },
        { id: "rtx-pro-6000",label: "NVIDIA RTX PRO 6000 Blackwell 96GB", gib: 96 },
        { id: "rtx-a6000",   label: "NVIDIA RTX A6000 48GB",          gib: 48  },
        { id: "rtx-5090",    label: "NVIDIA RTX 5090 32GB",           gib: 32  },
        { id: "rtx-4090",    label: "NVIDIA RTX 4090 24GB",           gib: 24  },
        { id: "mi300x-192",  label: "AMD Instinct MI300X 192GB",      gib: 192 },
        { id: "mi325x-256",  label: "AMD Instinct MI325X 256GB",      gib: 256 },
        { id: "mi355x-288",  label: "AMD Instinct MI355X 288GB",      gib: 288 }
    ];

    // ============================================================
    // Quick-pick HF examples (mix of public + public mirrors of gated repos)
    // ============================================================
    const HF_QUICKPICKS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
        "deepseek-ai/DeepSeek-V3",
        "unsloth/Llama-3.3-70B-Instruct",
        "unsloth/Llama-4-Scout-17B-16E-Instruct",
        "unsloth/gemma-3-27b-it"
    ];

    // ============================================================
    // Architecture-form fields. Keys here are the only thing the
    // override system tracks against — i.e. these are the fields that
    // get reset by "↺ reset" and stamped by presets.
    // ============================================================
    const ARCH_FIELDS = [
        "kvcc-attn", "kvcc-params", "kvcc-layers",
        "kvcc-qheads", "kvcc-kvheads", "kvcc-headdim",
        "kvcc-mla-lora", "kvcc-mla-rope",
        "kvcc-swa-window", "kvcc-swa-local", "kvcc-swa-global",
        "kvcc-hybrid-attn",
        "kvcc-wprec"
    ];

    // ============================================================
    // Helpers
    // ============================================================
    const $ = (id) => document.getElementById(id);
    const GiB = 1024 * 1024 * 1024;

    function fmtBytes(b) {
        if (!isFinite(b) || b < 0) return "—";
        if (b >= 1024 * GiB)   return (b / (1024 * GiB)).toFixed(2) + " TiB";
        if (b >= GiB)          return (b / GiB).toFixed(2) + " GiB";
        if (b >= 1024 * 1024)  return (b / (1024 * 1024)).toFixed(2) + " MiB";
        if (b >= 1024)         return (b / 1024).toFixed(2) + " KiB";
        return b.toFixed(0) + " B";
    }
    function fmtBytesSmall(b) {
        if (!isFinite(b) || b < 0) return "—";
        if (b >= GiB)          return (b / GiB).toFixed(3) + " GiB";
        if (b >= 1024 * 1024)  return (b / (1024 * 1024)).toFixed(2) + " MiB";
        if (b >= 1024)         return (b / 1024).toFixed(2) + " KiB";
        return b.toFixed(0) + " B";
    }

    function buildSelect(sel, items, mapper) {
        sel.innerHTML = items.map(mapper).join("");
    }

    // ============================================================
    // Override tracking
    // ============================================================
    /** @type {Set<string>} ids of fields the user has manually edited */
    const manualOverrides = new Set();

    /** @type {Object<string, string|number>} the last-applied baseline per field */
    let currentBaseline = {};

    function markEdited(fieldId) {
        if (!ARCH_FIELDS.includes(fieldId)) return;
        // Only mark when the value differs from the baseline. This stops
        // a re-emitted change event during applyBaseline() from registering.
        const el = $(fieldId);
        if (!el) return;
        const baseline = currentBaseline[fieldId];
        const current = el.value;
        if (baseline !== undefined && String(baseline) === String(current)) {
            manualOverrides.delete(fieldId);
        } else {
            manualOverrides.add(fieldId);
        }
        renderOverrideChips();
    }

    function renderOverrideChips() {
        // Toggle the data attribute that drives the "reset all" link visibility.
        $("kvcc").setAttribute(
            "data-has-overrides",
            manualOverrides.size > 0 ? "true" : "false"
        );

        // For each architecture field, ensure exactly one ↺ chip exists when
        // overridden, none when not.
        for (const id of ARCH_FIELDS) {
            const input = $(id);
            if (!input) continue;
            const label = input.parentElement.querySelector("label");
            if (!label) continue;
            const existing = label.querySelector(".kvcc-reset");
            const overridden = manualOverrides.has(id);

            if (overridden && !existing) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "kvcc-reset";
                btn.textContent = "↺ reset";
                btn.title = "Restore the value from the current preset baseline";
                btn.addEventListener("click", () => resetField(id));
                label.appendChild(btn);
            } else if (!overridden && existing) {
                existing.remove();
            }

            // Tooltip on hover shows the baseline value.
            if (existing && currentBaseline[id] !== undefined) {
                existing.title =
                    `Restore to baseline: ${currentBaseline[id]}`;
            }
        }
    }

    function resetField(fieldId) {
        if (currentBaseline[fieldId] === undefined) return;
        const el = $(fieldId);
        if (!el) return;
        el.value = currentBaseline[fieldId];
        manualOverrides.delete(fieldId);
        renderOverrideChips();
        recompute();
    }

    function resetAllOverrides() {
        for (const id of Array.from(manualOverrides)) {
            const el = $(id);
            if (el && currentBaseline[id] !== undefined) {
                el.value = currentBaseline[id];
            }
        }
        manualOverrides.clear();
        renderOverrideChips();
        recompute();
    }

    /**
     * Apply a baseline configuration to the form. Fields the user has
     * manually overridden are left untouched; the rest are stamped.
     *
     * @param {Object<string, string|number>} baseline keyed by field id
     */
    function applyBaseline(baseline) {
        currentBaseline = Object.assign({}, baseline);
        for (const [fieldId, value] of Object.entries(baseline)) {
            if (manualOverrides.has(fieldId)) continue;
            const el = $(fieldId);
            if (!el || value === undefined || value === null) continue;
            el.value = String(value);
        }
        // Attention type drives which conditional fields show/hide.
        $("kvcc").setAttribute("data-attn", $("kvcc-attn").value);
        renderOverrideChips();
    }

    // ============================================================
    // Built-in preset → baseline
    // ============================================================
    function presetToBaseline(m) {
        const b = {
            "kvcc-attn":    m.attn,
            "kvcc-params":  m.params,
            "kvcc-layers":  m.layers,
            "kvcc-qheads":  m.qHeads,
            "kvcc-kvheads": m.kvHeads,
            "kvcc-headdim": m.headDim
        };
        if (m.attn === "mla") {
            b["kvcc-mla-lora"] = m.mlaLora || 512;
            b["kvcc-mla-rope"] = m.mlaRope || 64;
        }
        if (m.attn === "swa") {
            b["kvcc-swa-window"] = m.swaWindow || 1024;
            b["kvcc-swa-local"]  = m.swaLocal  != null ? m.swaLocal  : 5;
            b["kvcc-swa-global"] = m.swaGlobal != null ? m.swaGlobal : 1;
        }
        if (m.attn === "hybrid") {
            b["kvcc-hybrid-attn"] =
                m.hybridAttn != null ? m.hybridAttn : Math.max(1, Math.round(m.layers / 4));
        }
        return b;
    }

    function applyModelPreset() {
        const id = $("kvcc-model").value;
        const m = MODELS.find((x) => x.id === id);
        if (!m) return;
        $("kvcc-model-hint").textContent = m.note || "";
        applyBaseline(presetToBaseline(m));
        // Hide the HF audit (we just switched away from an HF load).
        $("kvcc-hf-audit-wrap").hidden = true;
    }

    // ============================================================
    // Hugging Face loader
    // ============================================================
    const HF_BASE = "https://huggingface.co";
    const HF_TIMEOUT_MS = 8000;
    const HF_MAX_BYTES = 256 * 1024;
    const HF_REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

    function setHfStatus(kind, msg) {
        const el = $("kvcc-hf-status");
        el.className = "kvcc-hf-status";
        if (kind) el.classList.add("kvcc-hf-status--" + kind);
        el.textContent = msg || "";
    }

    async function fetchWithLimit(url) {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), HF_TIMEOUT_MS);
        try {
            const r = await fetch(url, {
                method: "GET",
                signal: ctrl.signal,
                headers: { "Accept": "application/json" }
            });
            // Handle non-2xx
            if (!r.ok) {
                const err = new Error(`HTTP ${r.status}`);
                err.status = r.status;
                throw err;
            }
            // Read up to HF_MAX_BYTES, then bail. We use the streaming reader
            // so we don't have to trust Content-Length.
            const reader = r.body.getReader();
            const chunks = [];
            let total = 0;
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                total += value.byteLength;
                if (total > HF_MAX_BYTES) {
                    ctrl.abort();
                    throw new Error("response exceeded 256 KB cap");
                }
                chunks.push(value);
            }
            const buf = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
            const text = new TextDecoder("utf-8").decode(buf);
            return JSON.parse(text);
        } finally {
            clearTimeout(timer);
        }
    }

    /**
     * Pick a numeric value out of an HF config, clamping to a sane range.
     */
    function clampInt(value, min, max, fallback) {
        const n = parseInt(value, 10);
        if (!isFinite(n)) return fallback;
        if (n < min) return min;
        if (n > max) return max;
        return n;
    }

    /**
     * Map a Hugging Face `config.json` (and api/models metadata) into our
     * internal "baseline" shape. Returns { baseline, info, warnings }.
     *
     * If the config nests transformer fields under `text_config` (Llama 4,
     * Gemma 3), we look there first.
     */
    function mapHfConfig(repoId, raw, meta) {
        // Some multi-modal configs nest the language-model arch.
        const cfg = raw.text_config && raw.text_config.num_hidden_layers
            ? raw.text_config
            : raw;

        const layers  = clampInt(cfg.num_hidden_layers,  1, 4096, 1);
        const qHeads  = clampInt(cfg.num_attention_heads, 1, 4096, 1);

        // KV head count: HF's official key is `num_key_value_heads`, but Falcon
        // (and a few older models) use `num_kv_heads`. Default to qHeads (i.e.
        // MHA) when neither is reported.
        const rawKv = cfg.num_key_value_heads != null
            ? cfg.num_key_value_heads
            : (cfg.num_kv_heads != null ? cfg.num_kv_heads : qHeads);
        const kvHeads = clampInt(rawKv, 1, 4096, qHeads);

        // Head dim: prefer explicit head_dim. For DeepSeek-style MLA configs,
        // use qk_nope_head_dim + qk_rope_head_dim if those are set (the
        // hidden_size/qHeads quotient doesn't reflect the actual per-token
        // attention head dimension). Otherwise fall back to hidden_size / qHeads.
        let headDim = clampInt(cfg.head_dim, 1, 4096, 0);
        if (!headDim && cfg.qk_nope_head_dim && cfg.qk_rope_head_dim != null) {
            headDim = clampInt(cfg.qk_nope_head_dim, 1, 4096, 0) +
                      clampInt(cfg.qk_rope_head_dim, 0, 4096, 0);
        }
        if (!headDim) {
            const hidden = clampInt(cfg.hidden_size, 1, 1 << 24, 0);
            if (hidden && qHeads) headDim = Math.floor(hidden / qHeads);
            if (!headDim) headDim = 128;
        }

        const warnings = [];
        const info = {
            modelType: cfg.model_type || raw.model_type,
            architectures: cfg.architectures || raw.architectures || []
        };

        // Detect attention type.
        let attn = "mha";
        const baseline = {
            "kvcc-layers":  layers,
            "kvcc-qheads":  qHeads,
            "kvcc-kvheads": kvHeads,
            "kvcc-headdim": headDim
        };

        if (cfg.kv_lora_rank != null) {
            // DeepSeek MLA family.
            attn = "mla";
            baseline["kvcc-mla-lora"] = clampInt(cfg.kv_lora_rank, 1, 65536, 512);
            baseline["kvcc-mla-rope"] = clampInt(
                cfg.qk_rope_head_dim != null ? cfg.qk_rope_head_dim : 64,
                0, 65536, 64
            );
        } else if (
            cfg.linear_attn_config ||
            cfg.linear_layer_indices ||
            cfg.linear_num_value_heads != null ||
            cfg.full_attention_interval != null ||
            cfg.attn_layer_indices != null ||
            (Array.isArray(cfg.layer_types) &&
                cfg.layer_types.some((t) =>
                    /linear|mamba|gated_delta|recurrent/.test(t)))
        ) {
            // Hybrid linear-state + attention (Qwen3-Next, Jamba, ...).
            attn = "hybrid";
            let attnLayers = 0;
            if (Array.isArray(cfg.attn_layer_indices)) {
                attnLayers = cfg.attn_layer_indices.length;
            } else if (Array.isArray(cfg.layer_types)) {
                attnLayers = cfg.layer_types.filter(
                    (t) => /^(full_)?attention$/i.test(t)
                ).length;
            } else if (cfg.full_attention_interval) {
                // Qwen3-Next style: every Nth layer is full attention.
                const interval = clampInt(cfg.full_attention_interval, 1, 1024, 4);
                attnLayers = Math.max(1, Math.floor(layers / interval));
            } else if (
                cfg.linear_attn_config &&
                typeof cfg.linear_attn_config === "object" &&
                cfg.linear_attn_config.linear_attn_period
            ) {
                // Fallback heuristic: 1-in-N-style schedules.
                const period = cfg.linear_attn_config.linear_attn_period;
                if (period >= 2) attnLayers = Math.max(1, Math.floor(layers / period));
            }
            if (!attnLayers) {
                attnLayers = Math.max(1, Math.round(layers / 4));
                warnings.push("Hybrid model: couldn't determine attention-layer " +
                              "count from config; assuming 1-in-4.");
            }
            baseline["kvcc-hybrid-attn"] = attnLayers;
        } else if (cfg.sliding_window && cfg.use_sliding_window !== false) {
            // Gemma-3-style local/global SWA, or pure SWA.
            const sw = clampInt(cfg.sliding_window, 1, 1 << 24, 1024);
            const pattern = clampInt(cfg.sliding_window_pattern, 0, 1024, 0);
            attn = "swa";
            baseline["kvcc-swa-window"] = sw;
            if (pattern && pattern > 1) {
                // sliding_window_pattern = N means 1 global per N-1 local
                // (HF convention used by Gemma 3).
                baseline["kvcc-swa-local"]  = pattern - 1;
                baseline["kvcc-swa-global"] = 1;
            } else {
                baseline["kvcc-swa-local"]  = 1;
                baseline["kvcc-swa-global"] = 0;
                warnings.push("Sliding-window detected but no global/local " +
                              "interleave reported; assuming pure SWA.");
            }
        } else if (kvHeads === 1) {
            attn = "mqa";
        } else if (kvHeads < qHeads) {
            attn = "gqa";
        } else {
            attn = "mha";
        }
        baseline["kvcc-attn"] = attn;

        // Weight precision: read torch_dtype if present.
        const dtype = (cfg.torch_dtype || raw.torch_dtype || "").toLowerCase();
        if (dtype === "bfloat16" || dtype === "float16" || dtype === "fp16" || dtype === "bf16") {
            baseline["kvcc-wprec"] = 2;
        } else if (dtype === "float8_e4m3fn" || dtype === "fp8" || dtype === "int8") {
            baseline["kvcc-wprec"] = 1;
        } else if (dtype === "int4") {
            baseline["kvcc-wprec"] = 0.5;
        }

        // Param count: prefer safetensors.total from the API.
        let paramsBytes = null;
        if (meta && meta.safetensors && typeof meta.safetensors.total === "number") {
            paramsBytes = meta.safetensors.total;
        }

        const wBytes = baseline["kvcc-wprec"] || 2;
        if (paramsBytes) {
            // safetensors.total counts ELEMENTS, not bytes for HF reports
            // historically — but the modern HF API actually returns parameter
            // count (count of tensor elements). Most repos return COUNT,
            // matching what we want directly.
            const billions = paramsBytes / 1e9;
            baseline["kvcc-params"] = Number(billions.toFixed(2));
            info.paramSource = "safetensors.total";
            info.params = billions;
        } else {
            // Fallback: if we don't know, leave whatever was there.
            warnings.push("safetensors.total not reported; param count not " +
                          "auto-set. Edit 'Total params (billions)' manually.");
            info.paramSource = "manual";
        }

        info.attn = attn;
        info.layers = layers;
        info.qHeads = qHeads;
        info.kvHeads = kvHeads;
        info.headDim = headDim;
        info.dtype = dtype || "(unspecified)";

        return { baseline, info, warnings };
    }

    async function loadFromHuggingFace(repoIdRaw) {
        const repoId = (repoIdRaw || "").trim();
        if (!repoId || !HF_REPO_RE.test(repoId)) {
            setHfStatus("err", "Invalid repo id. Expected `owner/model-name`.");
            return;
        }

        setHfStatus("load", `Fetching ${repoId}…`);
        $("kvcc-hf-load").disabled = true;

        try {
            const cfgUrl  = `${HF_BASE}/${repoId}/raw/main/config.json`;
            const metaUrl = `${HF_BASE}/api/models/${repoId}`;

            const [cfg, meta] = await Promise.all([
                fetchWithLimit(cfgUrl),
                fetchWithLimit(metaUrl).catch(() => null)
            ]);

            const { baseline, info, warnings } = mapHfConfig(repoId, cfg, meta);
            applyBaseline(baseline);

            // Reset the model-preset dropdown to "Custom" since we're on HF data.
            $("kvcc-model").value = "custom";
            $("kvcc-model-hint").textContent =
                `Loaded from Hugging Face: ${repoId}`;

            // Flip "Custom" preset's stored baseline so future "↺ reset"
            // restores the HF values, not the Custom defaults.
            // (applyBaseline already did this via currentBaseline.)

            // Build the audit panel.
            const auditWrap = $("kvcc-hf-audit-wrap");
            auditWrap.hidden = false;
            $("kvcc-hf-audit").textContent = renderHfAudit(repoId, info, warnings);

            const okMsg =
                `✓ Loaded ${repoId} — detected ${info.attn.toUpperCase()}, ` +
                `${info.layers} layers, ${info.qHeads}Q/${info.kvHeads}KV heads, ` +
                `head_dim ${info.headDim}` +
                (info.params ? `, ${info.params.toFixed(2)}B params` : "");
            setHfStatus(warnings.length ? "warn" : "ok",
                        okMsg + (warnings.length ? " (with warnings — see audit)" : ""));

            recompute();
        } catch (err) {
            console.error("HF load failed:", err);
            let msg = err.message || "fetch failed";
            if (err.status === 401) {
                msg = "401 Unauthorized — repo is gated or private. Try a " +
                      "public mirror like `unsloth/" + repoId.split("/")[1] + "`.";
            } else if (err.status === 404) {
                msg = "404 — repo not found. Check the spelling.";
            } else if (err.name === "AbortError") {
                msg = "request timed out.";
            }
            setHfStatus("err", "✗ " + msg);
        } finally {
            $("kvcc-hf-load").disabled = false;
        }
    }

    function renderHfAudit(repoId, info, warnings) {
        const lines = [];
        lines.push(`repo:          ${repoId}`);
        lines.push(`model_type:    ${info.modelType || "(unknown)"}`);
        lines.push(`architectures: ${(info.architectures || []).join(", ") || "(unknown)"}`);
        lines.push(``);
        lines.push(`detected attention type: ${info.attn.toUpperCase()}`);
        lines.push(`layers:    ${info.layers}`);
        lines.push(`Q heads:   ${info.qHeads}`);
        lines.push(`KV heads:  ${info.kvHeads}`);
        lines.push(`head_dim:  ${info.headDim}`);
        lines.push(`torch_dtype: ${info.dtype}`);
        lines.push(`param source: ${info.paramSource}`);
        if (info.params) lines.push(`params: ${info.params.toFixed(3)}B`);
        if (warnings.length) {
            lines.push(``);
            lines.push("warnings:");
            for (const w of warnings) lines.push("  - " + w);
        }
        return lines.join("\n");
    }

    // ============================================================
    // DOM bootstrap
    // ============================================================
    function init() {
        if (!$("kvcc")) return;

        buildSelect(
            $("kvcc-model"), MODELS,
            (m) => `<option value="${m.id}">${m.label}</option>`
        );
        buildSelect(
            $("kvcc-gpu"), GPUS,
            (g) => `<option value="${g.id}">${g.label}</option>`
        );

        // Quick-pick chips
        const quick = $("kvcc-hf-quick");
        quick.innerHTML = HF_QUICKPICKS.map(
            (id) => `<button type="button" class="kvcc-chip" data-repo="${id}">${id}</button>`
        ).join("");
        quick.addEventListener("click", (ev) => {
            const t = ev.target.closest(".kvcc-chip");
            if (!t) return;
            const repo = t.getAttribute("data-repo");
            $("kvcc-hf-id").value = repo;
            loadFromHuggingFace(repo);
        });

        // Sensible defaults
        $("kvcc-model").value = "llama-3.1-70b";
        $("kvcc-gpu").value   = "h100-80";

        applyModelPreset();
        bindEvents();
        recompute();
    }

    function bindEvents() {
        document.querySelectorAll(".kvcc-input").forEach((el) => {
            el.addEventListener("input", (ev) => {
                if (ARCH_FIELDS.includes(ev.target.id)) {
                    markEdited(ev.target.id);
                }
                recompute();
            });
            el.addEventListener("change", (ev) => {
                if (ARCH_FIELDS.includes(ev.target.id)) {
                    markEdited(ev.target.id);
                }
                recompute();
            });
        });

        $("kvcc-model").addEventListener("change", () => {
            applyModelPreset();
            recompute();
        });

        $("kvcc-attn").addEventListener("change", () => {
            $("kvcc").setAttribute("data-attn", $("kvcc-attn").value);
        });

        $("kvcc-gpu").addEventListener("change", recompute);

        $("kvcc-paged-eff").addEventListener("input", () => {
            $("kvcc-paged-eff-val").textContent = $("kvcc-paged-eff").value;
        });

        $("kvcc-hf-load").addEventListener("click", () => {
            loadFromHuggingFace($("kvcc-hf-id").value);
        });
        $("kvcc-hf-id").addEventListener("keydown", (ev) => {
            if (ev.key === "Enter") {
                ev.preventDefault();
                loadFromHuggingFace($("kvcc-hf-id").value);
            }
        });

        $("kvcc-reset-all").addEventListener("click", (ev) => {
            ev.preventDefault();
            resetAllOverrides();
        });
    }

    // ============================================================
    // Sizing math
    // ============================================================
    function readState() {
        const num  = (id) => parseFloat($(id).value) || 0;
        const intv = (id) => parseInt($(id).value, 10) || 0;

        const gpu = GPUS.find((g) => g.id === $("kvcc-gpu").value);

        return {
            attn:        $("kvcc-attn").value,
            params:      num("kvcc-params"),
            layers:      intv("kvcc-layers"),
            qHeads:      intv("kvcc-qheads"),
            kvHeads:     intv("kvcc-kvheads"),
            headDim:     intv("kvcc-headdim"),

            mlaLora:     intv("kvcc-mla-lora"),
            mlaRope:     intv("kvcc-mla-rope"),

            swaWindow:   intv("kvcc-swa-window"),
            swaLocal:    intv("kvcc-swa-local"),
            swaGlobal:   intv("kvcc-swa-global"),

            hybridAttn:  intv("kvcc-hybrid-attn"),

            seqLen:      intv("kvcc-seqlen"),
            batch:       intv("kvcc-batch"),
            kvBytes:     parseFloat($("kvcc-kvprec").value),
            wBytes:      parseFloat($("kvcc-wprec").value),
            overheadPct: num("kvcc-overhead"),
            fixedGiB:    num("kvcc-fixed-overhead"),
            prefixShare: intv("kvcc-prefix-share"),
            pagedEff:    Math.max(1, intv("kvcc-paged-eff")) / 100,

            gpuGib:      gpu ? gpu.gib : 0,
            gpuCount:    Math.max(1, intv("kvcc-gpu-count"))
        };
    }

    function computeKvBytesPerToken(s) {
        switch (s.attn) {
            case "mha":
            case "gqa":
            case "mqa":
                return {
                    perToken: 2 * s.layers * s.kvHeads * s.headDim * s.kvBytes,
                    totalBytesForTokens: (n) =>
                        2 * s.layers * s.kvHeads * s.headDim * s.kvBytes * n
                };
            case "mla": {
                const perLayer = (s.mlaLora + s.mlaRope) * s.kvBytes;
                return {
                    perToken: s.layers * perLayer,
                    totalBytesForTokens: (n) => s.layers * perLayer * n
                };
            }
            case "swa": {
                const cycle = Math.max(1, s.swaLocal + s.swaGlobal);
                const localLayers  = s.layers * (s.swaLocal  / cycle);
                const globalLayers = s.layers * (s.swaGlobal / cycle);
                const perTokHead   = 2 * s.kvHeads * s.headDim * s.kvBytes;
                const totalBytesForSeq = (T) => {
                    const localT = Math.min(T, s.swaWindow);
                    return perTokHead * (localLayers * localT + globalLayers * T);
                };
                return {
                    perToken: perTokHead *
                        (localLayers * Math.min(s.seqLen, s.swaWindow) +
                         globalLayers * s.seqLen) /
                        Math.max(1, s.seqLen),
                    totalBytesForSeq,
                    isSeqDependent: true
                };
            }
            case "hybrid": {
                const attnLayers = Math.min(s.hybridAttn, s.layers);
                return {
                    perToken: 2 * attnLayers * s.kvHeads * s.headDim * s.kvBytes,
                    totalBytesForTokens: (n) =>
                        2 * attnLayers * s.kvHeads * s.headDim * s.kvBytes * n
                };
            }
            default:
                return { perToken: 0, totalBytesForTokens: () => 0 };
        }
    }

    function computeTotalKvBytes(s) {
        const m = computeKvBytesPerToken(s);
        const prefix = Math.min(s.prefixShare, s.seqLen);
        const perSeqExtra = Math.max(0, s.seqLen - prefix);

        let raw;
        if (m.isSeqDependent) {
            const fullSeqBytes   = m.totalBytesForSeq(s.seqLen);
            const prefixSeqBytes = m.totalBytesForSeq(prefix);
            const extraBytes     = fullSeqBytes - prefixSeqBytes;
            raw = prefixSeqBytes + s.batch * extraBytes;
        } else {
            const totalTokens = prefix + s.batch * perSeqExtra;
            raw = m.totalBytesForTokens(totalTokens);
        }
        return raw / Math.max(0.01, s.pagedEff);
    }

    function compute() {
        const s = readState();
        const weightsBytes = s.params * 1e9 * s.wBytes;
        const kvTotalBytes = computeTotalKvBytes(s);

        const m = computeKvBytesPerToken(s);
        const kvPerToken = m.isSeqDependent
            ? (m.totalBytesForSeq(s.seqLen) / Math.max(1, s.seqLen))
            : m.perToken;

        const overheadBytes  = weightsBytes * (s.overheadPct / 100) + s.fixedGiB * GiB;
        const totalBytes     = weightsBytes + kvTotalBytes + overheadBytes;
        const availableBytes = s.gpuGib * s.gpuCount * GiB;

        return {
            s, weightsBytes, kvPerToken, kvTotalBytes,
            overheadBytes, totalBytes, availableBytes
        };
    }

    // ============================================================
    // Render
    // ============================================================
    function recompute() {
        const r = compute();

        $("kvcc-weights").textContent       = fmtBytes(r.weightsBytes);
        $("kvcc-kv-per-token").textContent  = fmtBytesSmall(r.kvPerToken);
        $("kvcc-kv-total").textContent      = fmtBytes(r.kvTotalBytes);
        $("kvcc-overhead-total").textContent = fmtBytes(r.overheadBytes);
        $("kvcc-total").textContent         = fmtBytes(r.totalBytes);
        $("kvcc-available").textContent     =
            r.availableBytes > 0 ? fmtBytes(r.availableBytes) : "(select a GPU)";
        const headroom = r.availableBytes - r.totalBytes;
        $("kvcc-headroom").textContent      =
            r.availableBytes > 0
                ? (headroom >= 0 ? "+" : "−") + fmtBytes(Math.abs(headroom))
                : "—";

        const cap = Math.max(r.availableBytes, r.totalBytes);
        $("kvcc-bar-weights").style.width  = ((r.weightsBytes  / cap) * 100).toFixed(2) + "%";
        $("kvcc-bar-kv").style.width       = ((r.kvTotalBytes  / cap) * 100).toFixed(2) + "%";
        $("kvcc-bar-overhead").style.width = ((r.overheadBytes / cap) * 100).toFixed(2) + "%";
        $("kvcc-bar-total").textContent    =
            fmtBytes(r.totalBytes) +
            (r.availableBytes > 0 ? " of " + fmtBytes(r.availableBytes) : "");

        const fit = $("kvcc-fit");
        fit.classList.remove("kvcc-fit--ok", "kvcc-fit--warn", "kvcc-fit--bad");
        if (r.availableBytes <= 0) {
            fit.textContent = "Select a GPU to see whether the workload fits.";
        } else if (r.totalBytes > r.availableBytes) {
            fit.classList.add("kvcc-fit--bad");
            fit.textContent =
                "✗ Will OOM: needs " + fmtBytes(r.totalBytes) +
                ", short by " + fmtBytes(r.totalBytes - r.availableBytes) + ".";
        } else if (r.totalBytes > r.availableBytes * 0.9) {
            fit.classList.add("kvcc-fit--warn");
            fit.textContent =
                "⚠ Tight fit: " + fmtBytes(r.totalBytes) +
                " of " + fmtBytes(r.availableBytes) + " — under 10% headroom.";
        } else {
            fit.classList.add("kvcc-fit--ok");
            fit.textContent =
                "✓ Fits: " + fmtBytes(r.totalBytes) +
                " of " + fmtBytes(r.availableBytes) +
                " (" + fmtBytes(r.availableBytes - r.totalBytes) + " free).";
        }

        $("kvcc-formula-text").textContent = renderFormula(r);
    }

    function renderFormula(r) {
        const s = r.s;
        const lines = [];
        lines.push("weights = params × bytes_per_param");
        lines.push("        = " + s.params + "B × " + s.wBytes +
                   "  =  " + fmtBytes(r.weightsBytes));
        lines.push("");

        switch (s.attn) {
            case "mha":
            case "gqa":
            case "mqa":
                lines.push("kv_per_token = 2 × layers × kv_heads × head_dim × kv_bytes");
                lines.push("             = 2 × " + s.layers + " × " + s.kvHeads + " × " +
                           s.headDim + " × " + s.kvBytes +
                           "  =  " + fmtBytesSmall(r.kvPerToken));
                break;
            case "mla":
                lines.push("kv_per_token (MLA) = layers × (kv_lora_rank + qk_rope_head_dim) × kv_bytes");
                lines.push("                   = " + s.layers + " × (" + s.mlaLora +
                           " + " + s.mlaRope + ") × " + s.kvBytes +
                           "  =  " + fmtBytesSmall(r.kvPerToken));
                break;
            case "swa": {
                const cycle = Math.max(1, s.swaLocal + s.swaGlobal);
                const localL  = s.layers * (s.swaLocal  / cycle);
                const globalL = s.layers * (s.swaGlobal / cycle);
                lines.push("local_layers = " + localL.toFixed(1) +
                           "   global_layers = " + globalL.toFixed(1));
                lines.push("kv_bytes(seq) = 2 × kv_heads × head_dim × kv_bytes ×");
                lines.push("                (local_layers × min(seq, window) + global_layers × seq)");
                lines.push("              = 2 × " + s.kvHeads + " × " + s.headDim +
                           " × " + s.kvBytes + " ×");
                lines.push("                (" + localL.toFixed(1) +
                           " × min(" + s.seqLen + ", " + s.swaWindow + ")");
                lines.push("                 + " + globalL.toFixed(1) +
                           " × " + s.seqLen + ")");
                break;
            }
            case "hybrid": {
                const a = Math.min(s.hybridAttn, s.layers);
                lines.push("kv_per_token (hybrid) = 2 × attention_layers × kv_heads × head_dim × kv_bytes");
                lines.push("                      = 2 × " + a + " × " + s.kvHeads +
                           " × " + s.headDim + " × " + s.kvBytes +
                           "  =  " + fmtBytesSmall(r.kvPerToken));
                break;
            }
        }

        const prefix = Math.min(s.prefixShare, s.seqLen);
        const perSeqExtra = Math.max(0, s.seqLen - prefix);
        lines.push("");
        lines.push("effective_tokens = prefix + batch × (seq − prefix)");
        lines.push("                 = " + prefix + " + " + s.batch + " × " + perSeqExtra +
                   " = " + (prefix + s.batch * perSeqExtra));
        lines.push("kv_total = kv_bytes(effective) / paged_efficiency");
        lines.push("         = " + fmtBytes(r.kvTotalBytes) +
                   "  (paged_eff = " + (s.pagedEff * 100).toFixed(0) + "%)");
        lines.push("");
        lines.push("overhead = weights × " + s.overheadPct + "%  +  " + s.fixedGiB + " GiB");
        lines.push("         = " + fmtBytes(r.overheadBytes));
        lines.push("");
        lines.push("TOTAL = weights + kv_total + overhead = " + fmtBytes(r.totalBytes));
        if (r.availableBytes > 0) {
            lines.push("AVAILABLE = " + fmtBytes(r.availableBytes) +
                       "   (" + s.gpuCount + " × " + s.gpuGib + " GiB)");
        }
        return lines.join("\n");
    }

    // Init
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
