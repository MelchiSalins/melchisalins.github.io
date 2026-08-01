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
 *   weights  = (params - quant_params) * bytes_per_param
 *              + quant_params * quant_bytes_per_param
 *              (quant_params = share stored quantized, e.g. MXFP4 MoE
 *               experts; 0 means uniform precision = params * bytes_per_param)
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

        // --- OpenAI gpt-oss (MXFP4 MoE experts, bf16 elsewhere) ---
        {
            id: "gpt-oss-120b",
            label: "gpt-oss 120B (SWA 1:1, MXFP4 MoE)",
            attn: "swa",
            params: 116.83, layers: 36, qHeads: 64, kvHeads: 8, headDim: 64,
            swaWindow: 128, swaLocal: 1, swaGlobal: 1,
            qParams: 114.66, qFormat: "mxfp4",
            note: "OpenAI gpt-oss-120b: 128-expert MoE, ~5.1B active. Experts " +
                  "ship MXFP4-quantized (~60.8 GiB total); attention/embeddings bf16. " +
                  "Alternating 128-token sliding / full attention."
        },
        {
            id: "gpt-oss-20b",
            label: "gpt-oss 20B (SWA 1:1, MXFP4 MoE)",
            attn: "swa",
            params: 20.91, layers: 24, qHeads: 64, kvHeads: 8, headDim: 64,
            swaWindow: 128, swaLocal: 1, swaGlobal: 1,
            qParams: 19.11, qFormat: "mxfp4",
            note: "OpenAI gpt-oss-20b: 32-expert MoE, ~3.6B active. MXFP4 experts " +
                  "(~12.8 GiB total), bf16 elsewhere. Alternating 128-token " +
                  "sliding / full attention."
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
    // GPU memory presets (nominal per-GPU capacity, treated as GiB)
    // Audited against vendor specifications on 2026-08-01. Preview values
    // are official projections and may change before general availability.
    // ============================================================
    const GPUS = [
        { id: "h100-80",     group: "NVIDIA Data Center", label: "NVIDIA H100 80GB (SXM/PCIe)",                    gib: 80  },
        { id: "h100-94",     group: "NVIDIA Data Center", label: "NVIDIA H100 NVL 94GB",                           gib: 94  },
        { id: "h200-141",    group: "NVIDIA Data Center", label: "NVIDIA H200 141GB",                              gib: 141 },
        { id: "b200-180",    group: "NVIDIA Data Center", label: "NVIDIA B200 180GB (HGX)",                        gib: 180 },
        { id: "gb200-186",   group: "NVIDIA Data Center", label: "NVIDIA GB200 186GB (per Blackwell GPU)",         gib: 186 },
        { id: "b300-270",    group: "NVIDIA Data Center", label: "NVIDIA B300 270GB (HGX)",                        gib: 270 },
        { id: "gb300-279",   group: "NVIDIA Data Center", label: "NVIDIA GB300 279GB (per Blackwell Ultra GPU)",   gib: 279 },
        { id: "rubin-288",   group: "NVIDIA Data Center", label: "NVIDIA Rubin 288GB (preliminary specification)", gib: 288 },
        { id: "a100-80",     group: "NVIDIA Data Center", label: "NVIDIA A100 80GB",                               gib: 80  },
        { id: "a100-40",     group: "NVIDIA Data Center", label: "NVIDIA A100 40GB",                               gib: 40  },
        { id: "l40s-48",     group: "NVIDIA Data Center", label: "NVIDIA L40S 48GB",                               gib: 48  },

        { id: "rtx-pro-6000",    group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX PRO 6000 Blackwell 96GB", gib: 96 },
        { id: "rtx-pro-5000-72", group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX PRO 5000 Blackwell 72GB", gib: 72 },
        { id: "rtx-pro-5000-48", group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX PRO 5000 Blackwell 48GB", gib: 48 },
        { id: "rtx-pro-4500-32", group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX PRO 4500 Blackwell 32GB", gib: 32 },
        { id: "rtx-pro-4000-24", group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX PRO 4000 Blackwell 24GB", gib: 24 },
        { id: "rtx-a6000",       group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX A6000 48GB",              gib: 48 },
        { id: "rtx-5090",        group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX 5090 32GB",               gib: 32 },
        { id: "rtx-4090",        group: "NVIDIA Professional / Consumer", label: "NVIDIA RTX 4090 24GB",               gib: 24 },

        { id: "mi300x-192", group: "AMD Instinct", label: "AMD Instinct MI300X 192GB", gib: 192 },
        { id: "mi325x-256", group: "AMD Instinct", label: "AMD Instinct MI325X 256GB", gib: 256 },
        { id: "mi350x-288", group: "AMD Instinct", label: "AMD Instinct MI350X 288GB", gib: 288 },
        { id: "mi355x-288", group: "AMD Instinct", label: "AMD Instinct MI355X 288GB", gib: 288 },

        { id: "radeon-ai-pro-r9700-32", group: "AMD Professional", label: "AMD Radeon AI PRO R9700 32GB", gib: 32 },

        { id: "arc-pro-b70-32", group: "Intel Professional", label: "Intel Arc Pro B70 32GB", gib: 32 },
        { id: "arc-pro-b65-32", group: "Intel Professional", label: "Intel Arc Pro B65 32GB", gib: 32 },
        { id: "arc-pro-b60-24", group: "Intel Professional", label: "Intel Arc Pro B60 24GB", gib: 24 },

        { id: "mi455x-432", group: "Preview (announced)", label: "AMD Instinct MI455X 432GB (Preview, expected 2H 2026)", gib: 432 },
        { id: "mi430x-432", group: "Preview (announced)", label: "AMD Instinct MI430X 432GB (Preview, expected 2027)",    gib: 432 }
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
        "unsloth/gemma-3-27b-it",
        "openai/gpt-oss-120b",
        "unsloth/kimi-k3"
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
        "kvcc-wprec", "kvcc-qparams", "kvcc-qformat"
    ];

    // Bytes-per-param for each quantized-format option (kvcc-qformat values).
    // MXFP4/MXFP8 include the E8M0 scale byte per 32-element group; NVFP4 the
    // FP8 scale per 16-element group.
    const QFORMAT = {
        mxfp4: 0.53125,   // 0.5 + 1/32
        nvfp4: 0.5625,    // 0.5 + 1/16
        int4:  0.5,
        fp8:   1,
        int8:  1,
        mxfp8: 1.03125,   // 1 + 1/32
        bf16:  2
    };

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

    function buildGroupedSelect(sel, items) {
        const groups = new Map();
        for (const item of items) {
            if (!groups.has(item.group)) groups.set(item.group, []);
            groups.get(item.group).push(item);
        }

        sel.replaceChildren();
        for (const [label, groupItems] of groups) {
            const group = document.createElement("optgroup");
            group.label = label;
            for (const item of groupItems) {
                const option = document.createElement("option");
                option.value = item.id;
                option.textContent = item.label;
                group.appendChild(option);
            }
            sel.appendChild(group);
        }
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
            "kvcc-headdim": m.headDim,
            // Always stamp the quantized split so switching away from a
            // quantized model (preset or HF) can't leave stale values.
            "kvcc-qparams": m.qParams || 0,
            "kvcc-qformat": QFORMAT[m.qFormat] || QFORMAT.mxfp4
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
     * Classify a repo's `quantization_config` into a family we can model.
     * Returns null when absent, else:
     *   { family, label, qFormat, groupSize, modeled, notes[] }
     * `modeled: false` means we recognised a quant method but can't derive
     * the byte layout reliably (GPTQ/AWQ/bnb pack shapes vary) — the user
     * must set the quantized split manually.
     */
    function detectQuantization(raw, cfg) {
        const qc = raw.quantization_config ||
                   (raw.text_config && raw.text_config.quantization_config) ||
                   cfg.quantization_config;
        if (!qc || typeof qc !== "object") return null;

        const method = String(qc.quant_method || "").toLowerCase();
        const notes = [];

        if (method === "mxfp4") {
            // gpt-oss style: MoE experts in MXFP4, rest bf16.
            return { family: "mxfp4", label: "mxfp4", qFormat: "mxfp4",
                     groupSize: 32, modeled: true, notes };
        }

        if (method === "fp8") {
            // DeepSeek style: FP8 weights + separate higher-precision scales.
            return { family: "fp8",
                     label: "fp8" + (qc.fmt ? " (" + qc.fmt + ")" : ""),
                     qFormat: "fp8", groupSize: 0, modeled: true, notes };
        }

        if (method === "compressed-tensors") {
            // Decide by the weight spec (type/num_bits/group_size), not the
            // format string — format vocabularies vary across repos.
            const groups = qc.config_groups && typeof qc.config_groups === "object"
                ? Object.values(qc.config_groups)
                : [];
            const specs = groups
                .map((g) => g && g.weights)
                .filter((w) => w && typeof w === "object");
            const label = "compressed-tensors" +
                (qc.format ? " (" + qc.format + ")" : "");
            if (!specs.length) {
                notes.push("compressed-tensors config without a weight spec; " +
                           "set the quantized split manually.");
                return { family: "compressed-tensors", label,
                         qFormat: null, groupSize: 0, modeled: false, notes };
            }
            if (specs.length > 1 &&
                specs.some((w) => w.num_bits !== specs[0].num_bits)) {
                notes.push("Multiple quantization groups with different bit " +
                           "widths; using the first group — verify manually.");
            }
            const w = specs[0];
            const bits = w.num_bits;
            const type = String(w.type || "").toLowerCase();
            const gs   = typeof w.group_size === "number" ? w.group_size : 0;
            if (type === "float" && bits === 4) {
                return gs === 16
                    ? { family: "nvfp4", label, qFormat: "nvfp4",
                        groupSize: 16, modeled: true, notes }
                    : { family: "mxfp4", label, qFormat: "mxfp4",
                        groupSize: gs || 32, modeled: true, notes };
            }
            if (type === "float" && bits === 8) {
                // Group-scaled FP8 = MXFP8 (scales ride along); otherwise
                // plain FP8 with separate scale tensors.
                return gs > 0
                    ? { family: "mxfp8", label, qFormat: "mxfp8",
                        groupSize: gs, modeled: true, notes }
                    : { family: "fp8", label, qFormat: "fp8",
                        groupSize: 0, modeled: true, notes };
            }
            if (type === "int" && bits === 8) {
                return { family: "int8", label, qFormat: "int8",
                         groupSize: gs, modeled: true, notes };
            }
            if (type === "int" && bits === 4) {
                return { family: "int4", label, qFormat: "int4",
                         groupSize: gs, modeled: true, notes };
            }
            notes.push("Unrecognised compressed-tensors weight spec (" +
                       type + "/" + bits + "-bit); set the split manually.");
            return { family: "compressed-tensors", label,
                     qFormat: null, groupSize: 0, modeled: false, notes };
        }

        // gptq / awq / bitsandbytes / torchao / hqq / ... — pack layouts
        // vary too much to derive bytes from the dtype map reliably.
        notes.push("Quantization method \"" + (method || "unknown") +
                   "\" detected but not modeled; set 'Quantized params' " +
                   "and format manually.");
        return { family: method || "unknown", label: method || "unknown",
                 qFormat: null, groupSize: 0, modeled: false, notes };
    }

    /**
     * Split the HF API per-dtype parameter counts into logical quantized vs
     * non-quantized params for a modeled quant family.
     *
     * HF convention (verified against gpt-oss-120b / kimi-k3, where the
     * derived bytes match the repos' safetensors index total_size exactly):
     * each packed-U8 "blocks" byte is counted as 2 params and each E8M0
     * scale byte as 1 param, so logical FP4 params = U8 × gs/(gs+1).
     */
    function splitQuantParams(quant, dtypeMap) {
        const gs = quant.groupSize || 32;
        let q = 0, nq = 0;
        for (const [k, v] of Object.entries(dtypeMap)) {
            if (typeof v !== "number" || !isFinite(v) || v <= 0) continue;
            const isF8 = k === "F8_E4M3" || k === "F8_E5M2";
            switch (quant.family) {
                case "mxfp4":
                    if (k === "U8") q += v * gs / (gs + 1);
                    else nq += v;
                    break;
                case "nvfp4":
                    // Packed FP4 pairs in U8; FP8 group scales live in the F8
                    // bucket and are already folded into qBytes — skip them.
                    if (k === "U8") q += v * 2;
                    else if (!isF8) nq += v;
                    break;
                case "fp8":
                    if (isF8) q += v;
                    else nq += v;
                    break;
                case "mxfp8":
                    if (isF8 || k === "U8") q += v;
                    else nq += v;
                    break;
                case "int8":
                    if (k === "I8" || k === "U8") q += v;
                    else nq += v;
                    break;
                case "int4":
                    if (k === "U8") q += v * 2;
                    else nq += v;
                    break;
                default:
                    nq += v;
            }
        }
        return { q, nq };
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
            // Hybrid MLA + linear attention (Kimi K3 / Kimi-Linear): only
            // the full-attention layers keep a (sequence-growing) MLA KV
            // cache; the linear-attention layers hold O(1) state. The
            // layers field feeds only the KV math, so stamp it with the
            // full-attention count.
            const fal = cfg.linear_attn_config &&
                        cfg.linear_attn_config.full_attn_layers;
            if (Array.isArray(fal) && fal.length > 0 && fal.length < layers) {
                baseline["kvcc-layers"] = fal.length;
                info.attnLayers  = fal.length;
                info.totalLayers = layers;
                warnings.push("Hybrid MLA + linear attention: " + fal.length +
                              " of " + layers + " layers use full (MLA) " +
                              "attention; the " + (layers - fal.length) +
                              " linear-attention layers hold O(1) state and " +
                              "are excluded from the KV cache. Layers set to " +
                              fal.length + ".");
            }
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
            } else if (cfg.linear_attn_config &&
                       Array.isArray(cfg.linear_attn_config.full_attn_layers)) {
                // Explicit index list of full-attention layers — exact.
                attnLayers = cfg.linear_attn_config.full_attn_layers.length;
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

            // Exact local/global counts from layer_types when it fully
            // describes the stack (gpt-oss: 18 sliding_attention + 18
            // full_attention). Only trust the counts when every entry is
            // one of the two — otherwise (e.g. Llama 4 chunked_attention)
            // fall through to the pattern/pure-SWA heuristics.
            let slidingCount = 0, fullCount = 0;
            if (Array.isArray(cfg.layer_types)) {
                slidingCount = cfg.layer_types.filter(
                    (t) => /sliding/i.test(t)).length;
                fullCount = cfg.layer_types.filter(
                    (t) => /^full_attention$/i.test(t)).length;
            }
            if (slidingCount > 0 &&
                slidingCount + fullCount === cfg.layer_types.length) {
                // The SWA formula consumes these as fractions of the total
                // layer count, so raw counts are exact for any ratio.
                baseline["kvcc-swa-local"]  = slidingCount;
                baseline["kvcc-swa-global"] = fullCount;
            } else if (pattern && pattern > 1) {
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

        // Quantization: read quantization_config (top-level or text_config).
        const quant = detectQuantization(raw, cfg);
        if (quant) {
            for (const n of quant.notes) warnings.push(n);
            info.quant = quant;
        }

        // Weight precision: read torch_dtype if present. This is the
        // precision of the NON-quantized share when a quant split is set.
        const dtype = (cfg.torch_dtype || raw.torch_dtype || "").toLowerCase();
        if (dtype === "bfloat16" || dtype === "float16" || dtype === "fp16" || dtype === "bf16") {
            baseline["kvcc-wprec"] = 2;
        } else if (dtype === "float8_e4m3fn" || dtype === "fp8" || dtype === "int8") {
            baseline["kvcc-wprec"] = 1;
        } else if (dtype === "int4") {
            baseline["kvcc-wprec"] = 0.5;
        } else if (quant && quant.modeled) {
            // Quantized repo without torch_dtype (gpt-oss): the un-quantized
            // modules (attention, embeddings, ...) are bf16.
            baseline["kvcc-wprec"] = 2;
        }

        // Param count: prefer the API's per-dtype map so quantized repos get
        // a logical split; fall back to safetensors.total.
        const st = meta && meta.safetensors ? meta.safetensors : null;
        const dtypeMap = st && st.parameters && typeof st.parameters === "object"
            ? st.parameters
            : null;

        // Always stamp the split so loading a non-quantized model after a
        // quantized one can't leave stale values.
        baseline["kvcc-qparams"] = 0;

        if (quant && quant.modeled && dtypeMap) {
            const { q, nq } = splitQuantParams(quant, dtypeMap);
            const totalB = (q + nq) / 1e9;
            const qB     = q / 1e9;
            baseline["kvcc-params"]  = Number(totalB.toFixed(2));
            baseline["kvcc-qparams"] = Number(qB.toFixed(2));
            baseline["kvcc-qformat"] = QFORMAT[quant.qFormat] || QFORMAT.mxfp4;
            info.paramSource = "safetensors.parameters (dtype split)";
            info.params  = totalB;
            info.qParams = qB;
            info.dtypeMap = dtypeMap;
        } else if (st && typeof st.total === "number") {
            // safetensors.total counts tensor ELEMENTS. For packed quantized
            // repos that inflates the logical count (2 params per packed U8
            // byte + scale bytes), which is why the dtype split above is
            // preferred when available.
            const billions = st.total / 1e9;
            baseline["kvcc-params"] = Number(billions.toFixed(2));
            info.paramSource = "safetensors.total";
            info.params = billions;
            if (quant && quant.modeled) {
                warnings.push("Quantized repo but the API returned no " +
                              "per-dtype parameter map; param count may be " +
                              "inflated and the quantized split was not set " +
                              "— adjust 'Quantized params' manually.");
            }
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

            const layersMsg = info.attnLayers
                ? `${info.attnLayers} attn layers (of ${info.totalLayers})`
                : `${info.layers} layers`;
            const okMsg =
                `✓ Loaded ${repoId} — detected ${info.attn.toUpperCase()}, ` +
                `${layersMsg}, ${info.qHeads}Q/${info.kvHeads}KV heads, ` +
                `head_dim ${info.headDim}` +
                (info.params ? `, ${info.params.toFixed(2)}B params` : "") +
                (info.quant && info.qParams
                    ? ` (${info.qParams.toFixed(1)}B ${info.quant.family})`
                    : "");
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
        if (info.attnLayers) {
            lines.push(`layers:    ${info.attnLayers} attention ` +
                       `(${info.totalLayers} total, ` +
                       `${info.totalLayers - info.attnLayers} linear-attention)`);
        } else {
            lines.push(`layers:    ${info.layers}`);
        }
        lines.push(`Q heads:   ${info.qHeads}`);
        lines.push(`KV heads:  ${info.kvHeads}`);
        lines.push(`head_dim:  ${info.headDim}`);
        lines.push(`torch_dtype: ${info.dtype}`);
        if (info.quant) {
            lines.push(``);
            lines.push(`quantization: ${info.quant.label}` +
                       (info.quant.groupSize
                           ? ` (group size ${info.quant.groupSize})` : ``) +
                       (info.quant.modeled ? `` : ` — not modeled`));
            if (info.dtypeMap) {
                const parts = Object.entries(info.dtypeMap)
                    .filter(([, v]) => typeof v === "number" && v > 0)
                    .map(([k, v]) => `${k} ${(v / 1e9).toFixed(2)}B`);
                lines.push(`safetensors dtypes: ${parts.join(", ")}`);
            }
        }
        lines.push(``);
        lines.push(`param source: ${info.paramSource}`);
        if (info.params) {
            if (info.qParams) {
                lines.push(`params: ${info.params.toFixed(3)}B total ` +
                           `(${info.qParams.toFixed(3)}B ${info.quant.family} ` +
                           `+ ${(info.params - info.qParams).toFixed(3)}B ` +
                           `higher-precision)`);
            } else {
                lines.push(`params: ${info.params.toFixed(3)}B`);
            }
        }
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
        buildGroupedSelect($("kvcc-gpu"), GPUS);

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
            qParams:     num("kvcc-qparams"),
            qBytes:      parseFloat($("kvcc-qformat").value),
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
        // Mixed precision: the quantized share (e.g. MXFP4 MoE experts) uses
        // qBytes/param, the rest (attention, embeddings, ...) uses wBytes.
        const qP = Math.min(s.qParams, s.params);
        const weightsBytes =
            (s.params - qP) * 1e9 * s.wBytes + qP * 1e9 * s.qBytes;
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
        const qP = Math.min(s.qParams, s.params);
        if (qP > 0) {
            lines.push("weights = (total − quant) × bytes_per_param + quant × quant_bytes");
            lines.push("        = " + (s.params - qP).toFixed(2) + "B × " + s.wBytes +
                       "  +  " + qP + "B × " + s.qBytes +
                       "  =  " + fmtBytes(r.weightsBytes));
        } else {
            lines.push("weights = params × bytes_per_param");
            lines.push("        = " + s.params + "B × " + s.wBytes +
                       "  =  " + fmtBytes(r.weightsBytes));
        }
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
