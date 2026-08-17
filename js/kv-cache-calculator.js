/* KV Cache Calculator — UI layer
 * ------------------------------
 * Pure-vanilla JS, no dependencies. All math runs in the browser.
 * The math + Hugging Face metadata interpretation lives in the model layer
 * (/js/kv-cache-model.js, loaded first; require()-able from Node for the
 * test-suite in tests/kv-cache/). This file owns DOM, state, and fetches.
 *
 * Two ways to populate the architecture fields:
 *   1. Pick a built-in preset from the "Model preset" dropdown.
 *   2. Type a Hugging Face repo id (e.g. `Qwen/Qwen2.5-7B-Instruct`) and
 *      click Load. We fetch only the few-KB `config.json` and the
 *      `/api/models/` metadata — never weights — and map the relevant
 *      fields onto the form. "Deep inspect" (optional, in the audit panel)
 *      additionally reads each shard's safetensors HEADER via HTTP Range
 *      requests (KBs–MBs, still never weights) for exact per-tensor truth.
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
 * Sizing math (see kv-cache-model.js for the formulas):
 *   weights_resident = (params − quant_params) × bytes_per_param
 *                      + quant_params × quant_bytes_per_param
 *   weights_disk     = checkpoint file sizes when known, else the
 *                      native-format equivalent (disk ≠ resident when the
 *                      quantized share is dequantized at load time)
 *   kv_total = growing KV (per attention type, paged) + O(1) recurrent
 *              state of linear-attention layers (per sequence, not paged)
 *   overhead = weights_resident × (overhead_pct/100) + fixed_overhead_GiB
 *   total    = weights_resident + kv_total + overhead
 */

(function () {
    "use strict";

    // Model layer (loaded via its own <script> tag before this file).
    const M = (typeof KVCCModel !== "undefined") ? KVCCModel : null;

    // ============================================================
    // Built-in model presets (popular configs, 2026 era)
    // `params` is total parameter count in billions. For MoE models that's
    // total params, since hosting requires loading every expert.
    // `qParams`/`qFormat` describe the checkpoint-native quantized share
    // (e.g. MXFP4 MoE experts); `diskGiB` is the shipped checkpoint size.
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

        // --- Moonshot Kimi (MLA, MoE; expert-only quantized checkpoints) ---
        {
            id: "kimi-k2.5",
            label: "Kimi K2.5 1T (MLA, MoE, INT4 experts)",
            attn: "mla",
            params: 1026.88, layers: 61, qHeads: 64, kvHeads: 64, headDim: 192,
            mlaLora: 512, mlaRope: 64,
            qParams: 1014.69, qFormat: "q4s", diskGiB: 554.3,
            note: "Moonshot Kimi K2.5 (DeepSeek-V3-style MLA, 384 routed " +
                  "experts). Routed experts ship INT4 (group-32 scales, " +
                  "compressed-tensors QAT); attention / shared expert / " +
                  "embeddings / vision tower stay BF16. ~554 GiB on disk."
        },
        {
            id: "kimi-k3",
            label: "Kimi K3 2.8T (MLA+KDA hybrid, MXFP4 experts)",
            attn: "mla",
            params: 2779.93, layers: 24, qHeads: 96, kvHeads: 96, headDim: 192,
            mlaLora: 512, mlaRope: 64,
            linearLayers: 69, stateKiB: 3360,
            qParams: 2722.74, qFormat: "mxfp4", diskGiB: 1453.7,
            note: "Moonshot Kimi K3 (unsloth mirror): 93 layers — 24 MLA " +
                  "full-attention (sequence-growing KV) + 69 KDA " +
                  "linear-attention layers holding ~3.3 MiB O(1) state per " +
                  "sequence each. Routed experts MXFP4; rest BF16. " +
                  "~1.42 TiB on disk."
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
            swaWindow: 128, swaLocal: 18, swaGlobal: 18,
            qParams: 114.66, qFormat: "mxfp4", diskGiB: 60.77,
            note: "OpenAI gpt-oss-120b: 128-expert MoE, ~5.1B active. Experts " +
                  "ship MXFP4-quantized (QAT — no BF16 original exists); " +
                  "attention/embeddings bf16. 18 sliding (128-token) + 18 " +
                  "full-attention layers. ~60.8 GiB on disk."
        },
        {
            id: "gpt-oss-20b",
            label: "gpt-oss 20B (SWA 1:1, MXFP4 MoE)",
            attn: "swa",
            params: 20.91, layers: 24, qHeads: 64, kvHeads: 8, headDim: 64,
            swaWindow: 128, swaLocal: 12, swaGlobal: 12,
            qParams: 19.11, qFormat: "mxfp4", diskGiB: 12.82,
            note: "OpenAI gpt-oss-20b: 32-expert MoE, ~3.6B active. MXFP4 " +
                  "experts, bf16 elsewhere. 12 sliding (128-token) + 12 " +
                  "full-attention layers. ~12.8 GiB on disk."
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
            hybridAttn: 12, linearLayers: 36, stateKiB: 1088,
            note: "Qwen3-Next: 12 Gated Attention + 36 Gated DeltaNet (3:1). " +
                  "80B total / 3B active. DeltaNet layers hold ~1.1 MiB O(1) " +
                  "state per sequence each."
        },
        {
            id: "jamba-1.5-mini",
            label: "Jamba 1.5 Mini 52B (Hybrid Mamba+Attn)",
            attn: "hybrid",
            params: 52, layers: 32, qHeads: 32, kvHeads: 8, headDim: 128,
            hybridAttn: 4, linearLayers: 28, stateKiB: 320,
            note: "AI21 Jamba 1.5 Mini. ~1:7 attention-to-Mamba ratio. " +
                  "52B total / 12B active. Mamba layers hold O(1) state."
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
        "unsloth/gemma-3-27b-it",
        "openai/gpt-oss-120b",
        "unsloth/Kimi-K3",
        "moonshotai/Kimi-K2.5"
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
        "kvcc-hybrid-attn", "kvcc-linear-layers", "kvcc-state-bytes",
        "kvcc-wprec", "kvcc-qparams", "kvcc-qformat"
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
    // Model-level metadata (checkpoint truth, not per-field form state)
    // ------------------------------------------------------------
    // Captured at preset/HF-load time; drives the disk-size row and the
    // Deep-inspect action. Reset on every model switch.
    //   diskBytes    — shipped checkpoint bytes (sibling sizes / headers)
    //   nativeQBytes — bytes/param of the checkpoint-native quantized
    //                  format (disk stays native even when the user picks
    //                  "Dequantized → BF16" for the resident format)
    //   repoId/shards/quant — Deep-inspect inputs (HF loads only)
    // ============================================================
    let modelMeta = { diskBytes: 0, nativeQBytes: 0,
                      repoId: null, shards: null, quant: null };

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
            // Always stamp the quantized/linear extras so switching away
            // from an exotic model can't leave stale values.
            "kvcc-qparams": m.qParams || 0,
            "kvcc-qformat": m.qFormat || "mxfp4",
            "kvcc-linear-layers": m.linearLayers || 0,
            "kvcc-state-bytes":   m.stateKiB || 0
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
        // Checkpoint-native metadata for the disk row.
        modelMeta = {
            diskBytes: (m.diskGiB || 0) * GiB,
            nativeQBytes: (m.qParams > 0 && M)
                ? M.qformatBytes(m.qFormat, 0) : 0,
            repoId: null, shards: null, quant: null
        };
        // Hide the HF audit and park Deep inspect (nothing to inspect until
        // a Hugging Face repo is loaded).
        $("kvcc-hf-audit-wrap").hidden = true;
        setDeepInspectReady(false);
        $("kvcc-deep-status").textContent = "";
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

    // Deep inspect is always visible next to Load; its disabled state +
    // tooltip tell the user whether it's actionable yet.
    const DEEP_TITLE_DISABLED =
        "Load a Hugging Face repo first";
    const DEEP_TITLE_READY =
        "Reads each shard's safetensors header via HTTP Range requests — " +
        "exact per-tensor numbers, slower on many-shard repos, never weights";

    function setDeepInspectReady(ready) {
        const btn = $("kvcc-deep-inspect");
        btn.disabled = !ready;
        btn.title = ready ? DEEP_TITLE_READY : DEEP_TITLE_DISABLED;
    }

    async function fetchWithLimit(url, maxBytes, timeoutMs) {
        const cap = maxBytes || HF_MAX_BYTES;
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), timeoutMs || HF_TIMEOUT_MS);
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
            // Read up to `cap` bytes, then bail. We use the streaming reader
            // so we don't have to trust Content-Length.
            const reader = r.body.getReader();
            const chunks = [];
            let total = 0;
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                total += value.byteLength;
                if (total > cap) {
                    ctrl.abort();
                    throw new Error("response exceeded " +
                                    Math.round(cap / 1024) + " KB cap");
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
     * Fetch an exact byte range [start, end] (inclusive) of a URL.
     * Prefers a Range request (206); degrades to a plain streaming read
     * with skip + early-abort when the server ignores Range (200) or the
     * CORS preflight refuses the Range header entirely (TypeError). The
     * degraded path stays cheap for our use case because safetensors
     * headers live at the START of the file.
     */
    async function fetchRange(url, start, end, timeoutMs) {
        const want = end - start + 1;

        const attempt = async (useRange) => {
            const ctrl = new AbortController();
            const timer = setTimeout(() => ctrl.abort(), timeoutMs || 20000);
            try {
                const r = await fetch(url, {
                    method: "GET",
                    signal: ctrl.signal,
                    headers: useRange ? { "Range": `bytes=${start}-${end}` } : {}
                });
                if (!r.ok && r.status !== 206) {
                    const err = new Error(`HTTP ${r.status}`);
                    err.status = r.status;
                    throw err;
                }
                const ranged = useRange && r.status === 206;
                const reader = r.body.getReader();
                const buf = new Uint8Array(want);
                let filled = 0;      // bytes written into buf
                let skipped = 0;     // bytes skipped when reading from 0
                const skipTarget = ranged ? 0 : start;
                while (filled < want) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    let chunk = value;
                    if (skipped < skipTarget) {
                        const rem = skipTarget - skipped;
                        if (chunk.byteLength <= rem) {
                            skipped += chunk.byteLength;
                            continue;
                        }
                        chunk = chunk.subarray(rem);
                        skipped = skipTarget;
                    }
                    const take = Math.min(chunk.byteLength, want - filled);
                    buf.set(chunk.subarray(0, take), filled);
                    filled += take;
                }
                ctrl.abort(); // stop the stream, we have what we need
                if (filled < want) {
                    throw new Error("short read (" + filled + "/" + want + " bytes)");
                }
                return buf;
            } finally {
                clearTimeout(timer);
            }
        };

        try {
            return await attempt(true);
        } catch (err) {
            // TypeError = network/CORS failure (e.g. preflight rejected the
            // Range header). Retry once as a plain read; anything else
            // (HTTP status, short read, abort) propagates.
            if (err instanceof TypeError) return attempt(false);
            throw err;
        }
    }

    async function loadFromHuggingFace(repoIdRaw) {
        const repoId = (repoIdRaw || "").trim();
        if (!repoId || !HF_REPO_RE.test(repoId)) {
            setHfStatus("err", "Invalid repo id. Expected `owner/model-name`.");
            return;
        }
        if (!M) {
            setHfStatus("err", "Model script failed to load; reload the page.");
            return;
        }

        setHfStatus("load", `Fetching ${repoId}…`);
        $("kvcc-hf-load").disabled = true;

        try {
            const cfgUrl  = `${HF_BASE}/${repoId}/raw/main/config.json`;
            // blobs=true adds per-file sizes → checkpoint-on-disk truth.
            const metaUrl = `${HF_BASE}/api/models/${repoId}?blobs=true`;

            const [cfg, meta] = await Promise.all([
                fetchWithLimit(cfgUrl),
                fetchWithLimit(metaUrl, 1024 * 1024).catch(() => null)
            ]);

            const { baseline, info, warnings } =
                M.buildHfBaseline(repoId, cfg, meta);
            applyBaseline(baseline);

            // Checkpoint-native metadata (disk row + deep inspect).
            const shards = meta && Array.isArray(meta.siblings)
                ? meta.siblings
                    .map((s) => s && s.rfilename)
                    .filter((f) => typeof f === "string" &&
                                   /^model[^/]*\.safetensors$/.test(f))
                : null;
            modelMeta = {
                diskBytes: info.diskBytes || 0,
                nativeQBytes: info.quant && info.quant.qBytesPerParam
                    ? info.quant.qBytesPerParam : 0,
                repoId,
                shards: shards && shards.length ? shards : null,
                quant: info.quant || null
            };

            // Reset the model-preset dropdown to "Custom" since we're on HF data.
            $("kvcc-model").value = "custom";
            $("kvcc-model-hint").textContent =
                `Loaded from Hugging Face: ${repoId}`;

            // Build the audit panel.
            const auditWrap = $("kvcc-hf-audit-wrap");
            auditWrap.hidden = false;
            $("kvcc-hf-audit").textContent = renderHfAudit(repoId, info, warnings);
            setDeepInspectReady(!!modelMeta.shards);
            $("kvcc-deep-status").textContent = modelMeta.shards
                ? "" : "(shard list unavailable — deep inspect disabled)";

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
                    : "") +
                (info.diskBytes ? `, ${fmtBytes(info.diskBytes)} on disk` : "");
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
        if (info.nestedKey) {
            lines.push(`lm config:     nested under \`${info.nestedKey}\``);
        }
        lines.push(``);
        lines.push(`detected attention type: ${info.attn.toUpperCase()}`);
        if (info.attnLayers) {
            lines.push(`layers:    ${info.attnLayers} attention ` +
                       `(${info.totalLayers} total, ` +
                       `${info.totalLayers - info.attnLayers} linear-attention)`);
            if (info.stateBytesPerLayer) {
                lines.push(`linear state: ~` +
                           fmtBytesSmall(info.stateBytesPerLayer) +
                           `/layer/sequence (${info.stateSource})`);
            }
        } else {
            lines.push(`layers:    ${info.layers}`);
        }
        lines.push(`Q heads:   ${info.qHeads}`);
        lines.push(`KV heads:  ${info.kvHeads}`);
        lines.push(`head_dim:  ${info.headDim}`);
        lines.push(`dtype:     ${info.dtype}`);
        if (info.quant) {
            lines.push(``);
            lines.push(`quantization: ${info.quant.label}` +
                       (info.quant.groupSize
                           ? ` (group size ${info.quant.groupSize})` : ``) +
                       (info.quant.modeled ? `` : ` — not modeled`));
            if (info.quant.qBytesPerParam) {
                lines.push(`quantized share: ${info.quant.qBytesPerParam} B/param ` +
                           `(scale overhead included)`);
            }
            if (info.quant.excludePatterns && info.quant.excludePatterns.length) {
                lines.push(`kept high-precision: ` +
                           info.quant.excludePatterns.join(", "));
            }
            if (info.kvCacheScheme) {
                lines.push(`kv_cache_scheme: ` +
                           JSON.stringify(info.kvCacheScheme));
            }
            if (info.dtypeMap) {
                const parts = Object.entries(info.dtypeMap)
                    .filter(([, v]) => typeof v === "number" && v > 0)
                    .map(([k, v]) => `${k} ${(v / 1e9).toFixed(2)}B`);
                lines.push(`safetensors dtypes: ${parts.join(", ")}`);
            }
            if (info.split) {
                lines.push(`histogram convention: ${info.split.convention}` +
                           (info.split.diskDeltaPct != null
                               ? ` (disk reconciliation Δ ` +
                                 info.split.diskDeltaPct.toFixed(2) + `%)`
                               : ``));
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
        if (info.diskBytes) {
            lines.push(`checkpoint on disk: ${fmtBytes(info.diskBytes)} ` +
                       `(root *.safetensors shards)`);
        }
        if (warnings.length) {
            lines.push(``);
            lines.push("warnings:");
            for (const w of warnings) lines.push("  - " + w);
        }
        return lines.join("\n");
    }

    // ============================================================
    // Deep inspect (Tier 4): read safetensors shard HEADERS via HTTP
    // Range requests — exact per-tensor dtype/shape/bytes, never weights.
    // ============================================================
    const INSPECT_MAX_SHARDS = 160;
    const INSPECT_HEADER_CAP = 64 * 1024 * 1024;  // total across shards
    const INSPECT_CONCURRENCY = 4;

    async function deepInspect() {
        if (!M || !modelMeta.repoId || !modelMeta.shards) return;
        const btn = $("kvcc-deep-inspect");
        const status = $("kvcc-deep-status");
        btn.disabled = true;

        const shards = modelMeta.shards.slice(0, INSPECT_MAX_SHARDS);
        const skipped = modelMeta.shards.length - shards.length;
        const acc = M.makeInspectAccumulator();
        let headerBytes = 0;
        let done = 0;
        let failed = null;

        const worker = async (queue) => {
            while (queue.length && !failed) {
                const file = queue.shift();
                const url = `${HF_BASE}/${modelMeta.repoId}/resolve/main/${file}`;
                try {
                    // safetensors layout: u64-LE header length, then the
                    // JSON header, then tensor data.
                    const lenBuf = await fetchRange(url, 0, 7);
                    const len = Number(new DataView(
                        lenBuf.buffer, lenBuf.byteOffset, 8
                    ).getBigUint64(0, true));
                    if (!isFinite(len) || len <= 0 || len > 128 * 1024 * 1024) {
                        throw new Error("implausible header length " + len);
                    }
                    headerBytes += len;
                    if (headerBytes > INSPECT_HEADER_CAP) {
                        throw new Error("header byte budget exceeded");
                    }
                    const hdrBuf = await fetchRange(url, 8, 8 + len - 1);
                    const header = JSON.parse(
                        new TextDecoder("utf-8").decode(hdrBuf));
                    M.accumulateHeader(acc, header, modelMeta.quant);
                    done++;
                    status.textContent =
                        `Inspecting shard headers… ${done}/${shards.length}`;
                } catch (e) {
                    failed = e;
                }
            }
        };

        try {
            status.textContent = `Inspecting shard headers… 0/${shards.length}`;
            const queue = shards.slice();
            await Promise.all(
                Array.from({ length: INSPECT_CONCURRENCY }, () => worker(queue)));
            if (failed) throw failed;

            // Apply exact numbers: params + quantized split + disk bytes.
            const totalB = (acc.qParams + acc.nqParams) / 1e9;
            const qB     = acc.qParams / 1e9;
            $("kvcc-params").value  = totalB.toFixed(2);
            $("kvcc-qparams").value = qB.toFixed(2);
            currentBaseline["kvcc-params"]  = Number(totalB.toFixed(2));
            currentBaseline["kvcc-qparams"] = Number(qB.toFixed(2));
            manualOverrides.delete("kvcc-params");
            manualOverrides.delete("kvcc-qparams");
            renderOverrideChips();
            modelMeta.diskBytes = acc.totalBytes;

            // Report into the audit panel.
            const lines = [];
            lines.push("");
            lines.push("deep inspect (safetensors headers, " +
                       done + " shard" + (done === 1 ? "" : "s") +
                       (skipped > 0 ? ", " + skipped + " skipped" : "") + "):");
            const dt = Object.entries(acc.byDtype)
                .sort((a, b) => b[1].bytes - a[1].bytes)
                .map(([k, v]) => `  ${k}: ${(v.elems / 1e9).toFixed(3)}B elems, ` +
                                 `${fmtBytes(v.bytes)} (${v.tensors} tensors)`);
            lines.push(...dt);
            lines.push(`  logical params: ${totalB.toFixed(3)}B total, ` +
                       `${qB.toFixed(3)}B quantized`);
            lines.push(`  group scales:   ${fmtBytes(acc.scaleBytes)}`);
            lines.push(`  checkpoint payload: ${fmtBytes(acc.totalBytes)}`);
            $("kvcc-hf-audit").textContent += "\n" + lines.join("\n");

            status.textContent =
                `✓ Verified from headers: ${totalB.toFixed(2)}B params, ` +
                `${fmtBytes(acc.totalBytes)} on disk.`;
            recompute();
        } catch (err) {
            console.error("deep inspect failed:", err);
            status.textContent = "✗ Deep inspect failed: " +
                (err && err.message ? err.message : "network/CORS error");
        } finally {
            btn.disabled = false;
        }
    }

    // ============================================================
    // DOM bootstrap
    // ============================================================
    function init() {
        if (!$("kvcc")) return;
        if (!M) {
            console.error("kv-cache-model.js missing; calculator disabled.");
            return;
        }

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

        $("kvcc-deep-inspect").addEventListener("click", deepInspect);

        $("kvcc-reset-all").addEventListener("click", (ev) => {
            ev.preventDefault();
            resetAllOverrides();
        });
    }

    // ============================================================
    // Sizing math (delegates to the model layer)
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
            linearLayers: intv("kvcc-linear-layers"),
            stateBytes:  num("kvcc-state-bytes") * 1024, // field unit: KiB

            seqLen:      intv("kvcc-seqlen"),
            batch:       intv("kvcc-batch"),
            kvBytes:     parseFloat($("kvcc-kvprec").value),
            wBytes:      parseFloat($("kvcc-wprec").value),
            qParams:     num("kvcc-qparams"),
            qFormat:     $("kvcc-qformat").value,
            qBytes:      M.qformatBytes($("kvcc-qformat").value, 0.53125),
            overheadPct: num("kvcc-overhead"),
            fixedGiB:    num("kvcc-fixed-overhead"),
            prefixShare: intv("kvcc-prefix-share"),
            pagedEff:    Math.max(1, intv("kvcc-paged-eff")) / 100,

            gpuGib:      gpu ? gpu.gib : 0,
            gpuCount:    Math.max(1, intv("kvcc-gpu-count"))
        };
    }

    function compute() {
        const s = readState();
        const w = M.computeWeightBytes(s, modelMeta);
        const kv = M.computeTotalKvBytes(s);

        const m = computeKvPerToken(s);
        const overheadBytes  = w.residentBytes * (s.overheadPct / 100) +
                               s.fixedGiB * GiB;
        const totalBytes     = w.residentBytes + kv.kvBytes + overheadBytes;
        const availableBytes = s.gpuGib * s.gpuCount * GiB;

        return {
            s,
            weightsBytes: w.residentBytes,
            diskBytes: w.diskBytes,
            diskSource: w.diskSource,
            qP: w.qP,
            kvPerToken: m,
            kvTotalBytes: kv.kvBytes,
            kvPagedBytes: kv.pagedKvBytes,
            kvStateBytes: kv.stateBytes,
            overheadBytes, totalBytes, availableBytes
        };
    }

    function computeKvPerToken(s) {
        const m = M.computeKvBytesPerToken(s);
        return m.isSeqDependent
            ? (m.totalBytesForSeq(s.seqLen) / Math.max(1, s.seqLen))
            : m.perToken;
    }

    // ============================================================
    // Render
    // ============================================================
    function recompute() {
        const r = compute();

        $("kvcc-weights").textContent       = fmtBytes(r.weightsBytes);
        const diskCell = $("kvcc-disk");
        if (diskCell) {
            const differs = Math.abs(r.diskBytes - r.weightsBytes) >
                            Math.max(1e6, r.weightsBytes * 0.005);
            diskCell.textContent = fmtBytes(r.diskBytes) +
                (differs ? "" : " (≈ resident)");
        }
        $("kvcc-kv-per-token").textContent  = fmtBytesSmall(r.kvPerToken);
        $("kvcc-kv-total").textContent      = fmtBytes(r.kvTotalBytes) +
            (r.kvStateBytes > 0
                ? " (incl. " + fmtBytes(r.kvStateBytes) + " linear state)"
                : "");
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
        const qP = r.qP;
        if (qP > 0) {
            lines.push("weights = (total − quant) × bytes_per_param + quant × quant_bytes");
            lines.push("        = " + (s.params - qP).toFixed(2) + "B × " + s.wBytes +
                       "  +  " + qP + "B × " + s.qBytes +
                       "  =  " + fmtBytes(r.weightsBytes) + "  (resident in VRAM)");
        } else {
            lines.push("weights = params × bytes_per_param");
            lines.push("        = " + s.params + "B × " + s.wBytes +
                       "  =  " + fmtBytes(r.weightsBytes) + "  (resident in VRAM)");
        }
        lines.push("on disk = " + fmtBytes(r.diskBytes) +
                   "  (" + r.diskSource + ")");
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
        lines.push("kv_growing = kv_bytes(effective) / paged_efficiency");
        lines.push("           = " + fmtBytes(r.kvPagedBytes) +
                   "  (paged_eff = " + (s.pagedEff * 100).toFixed(0) + "%)");
        if (r.kvStateBytes > 0) {
            lines.push("linear_state = linear_layers × state/layer × batch");
            lines.push("             = " + s.linearLayers + " × " +
                       fmtBytesSmall(s.stateBytes) + " × " + Math.max(1, s.batch) +
                       "  =  " + fmtBytes(r.kvStateBytes) +
                       "  (O(1) per sequence, not paged)");
            lines.push("kv_total = kv_growing + linear_state = " +
                       fmtBytes(r.kvTotalBytes));
        } else {
            lines.push("kv_total = " + fmtBytes(r.kvTotalBytes));
        }
        lines.push("");
        lines.push("overhead = weights_resident × " + s.overheadPct + "%  +  " +
                   s.fixedGiB + " GiB");
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
