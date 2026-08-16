/* KV Cache Calculator — model layer
 * ---------------------------------
 * Pure logic, no DOM. Loaded by the browser before kv-cache-calculator.js
 * (which owns all DOM/fetch orchestration) and require()-able from Node for
 * the test-suite in tests/kv-cache/.
 *
 * Source-of-truth ladder (every derived number carries a `source` tag):
 *   1. config.json  — HF Transformers schema: architecture geometry,
 *      layer_types, quantization_config (dispatch on `quant_method`, the
 *      same discriminator transformers' quantizer registry dispatches on),
 *      dtype / torch_dtype.
 *   2. quantization_config payloads — compressed-tensors config_groups /
 *      targets / ignore / kv_cache_scheme, transformers-mxfp4
 *      modules_to_not_convert.
 *   3. HF Hub API — safetensors.parameters per-dtype histogram + sibling
 *      file sizes (disk-byte ground truth for the shipped checkpoint).
 *   4. safetensors shard headers (Tier-4 "deep inspect") — exact per-tensor
 *      dtype/shape, fetched via HTTP Range by the UI layer and classified
 *      here.
 *
 * The HF param histogram is NOT stable across time/CDN generations: the
 * same repo has been observed reporting packed-element counts including
 * group scales (old) and logical parameter counts excluding scales (new).
 * interpretDtypeMap() therefore scores both interpretations against the
 * checkpoint's actual on-disk bytes and picks the one that reconciles.
 */

(function (root) {
    "use strict";

    // ============================================================
    // Precision-format registry
    // ------------------------------------------------------------
    // bytesPerParam includes group-scale overhead where the scales ride
    // along with the packed weights (MX formats store one E8M0 scale per
    // group; compressed-tensors INT4 typically one FP16 scale per group).
    // ============================================================
    const QFORMAT = {
        mxfp4: { bytes: 0.53125,  label: "MXFP4 (4-bit + E8M0 scale /32)" },
        nvfp4: { bytes: 0.5625,   label: "NVFP4 (4-bit + FP8 scale /16)" },
        q4s:   { bytes: 0.5625,   label: "INT4 + 16-bit group scales /32" },
        int4:  { bytes: 0.5,      label: "INT4 packed (no scale overhead)" },
        fp8:   { bytes: 1,        label: "FP8 (separate scale tensors)" },
        int8:  { bytes: 1,        label: "INT8" },
        mxfp8: { bytes: 1.03125,  label: "MXFP8 (8-bit + E8M0 scale /32)" },
        bf16:  { bytes: 2,        label: "BF16 (dequantized)" }
    };

    function qformatBytes(id, fallback) {
        return QFORMAT[id] ? QFORMAT[id].bytes
                           : (fallback != null ? fallback : QFORMAT.mxfp4.bytes);
    }

    /** Exact bytes/param for a parsed weight spec, scale overhead included. */
    function weightSpecBytes(bits, groupSize, scaleBytes) {
        const payload = bits / 8;
        if (!groupSize || groupSize <= 0) return payload;
        return payload + (scaleBytes || 0) / groupSize;
    }

    // safetensors dtype strings → bytes per element.
    const DTYPE_BYTES = {
        F64: 8, I64: 8, U64: 8,
        F32: 4, I32: 4, U32: 4,
        BF16: 2, F16: 2, I16: 2, U16: 2,
        F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1,
        I8: 1, U8: 1, BOOL: 1,
        F4: 0.5, F6_E2M3: 0.75, F6_E3M2: 0.75
    };

    function dtypeBytes(k) {
        return DTYPE_BYTES[String(k).toUpperCase()] != null
            ? DTYPE_BYTES[String(k).toUpperCase()]
            : 2;
    }

    // ============================================================
    // Small helpers
    // ============================================================
    function clampInt(value, min, max, fallback) {
        const n = parseInt(value, 10);
        if (!isFinite(n)) return fallback;
        if (n < min) return min;
        if (n > max) return max;
        return n;
    }

    /**
     * Match a module name against quantization module patterns.
     * Supports both dialects seen in the wild:
     *   - compressed-tensors `ignore`: "re:<regex>" entries
     *   - transformers `modules_to_not_convert`: fnmatch-style globs
     *     ("model.layers.*.self_attn") or bare substrings.
     */
    function moduleMatches(name, patterns) {
        if (!Array.isArray(patterns)) return false;
        for (const p of patterns) {
            if (typeof p !== "string" || !p) continue;
            try {
                if (p.startsWith("re:")) {
                    if (new RegExp(p.slice(3)).test(name)) return true;
                } else if (p.indexOf("*") !== -1) {
                    const rx = "^" + p.split("*").map(escapeRe).join(".*") + "$";
                    if (new RegExp(rx).test(name) ||
                        new RegExp(rx.slice(1, -1)).test(name)) return true;
                } else if (name === p || name.startsWith(p + ".") ||
                           name.indexOf("." + p + ".") !== -1 ||
                           name.endsWith("." + p)) {
                    return true;
                }
            } catch (e) { /* ignore malformed pattern */ }
        }
        return false;
    }

    function escapeRe(s) {
        return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    /**
     * Locate the language-model config inside a (possibly multimodal)
     * config.json. Transformers nests it under different keys per family.
     */
    function findLmConfig(raw) {
        if (!raw || typeof raw !== "object") return { cfg: raw || {}, nestedKey: null };
        const keys = ["text_config", "language_config", "llm_config", "decoder"];
        for (const k of keys) {
            const c = raw[k];
            if (c && typeof c === "object" && c.num_hidden_layers) {
                return { cfg: c, nestedKey: k };
            }
        }
        return { cfg: raw, nestedKey: null };
    }

    // ============================================================
    // Quantization detection (Tier 2)
    // ============================================================
    /**
     * Classify a repo's `quantization_config` into a family we can model.
     * Scans the top level and known nesting keys — Kimi-K2.5 ships it only
     * inside `text_config`, gpt-oss only at the top, Kimi-K3 in both.
     *
     * Returns null when absent, else:
     *   { family, label, qFormat, qBytesPerParam, groupSize, bits,
     *     excludePatterns[], targets[], kvCacheScheme, activations,
     *     modeled, notes[] }
     * `modeled: false` = recognised but byte layout is not derivable
     * (GPTQ/AWQ/bnb pack shapes vary) — the split must be set manually,
     * or verified with Deep inspect.
     */
    function detectQuantization(raw) {
        const spots = [raw,
                       raw && raw.text_config,
                       raw && raw.language_config,
                       raw && raw.llm_config];
        let qc = null;
        for (const s of spots) {
            if (s && typeof s === "object" &&
                s.quantization_config && typeof s.quantization_config === "object") {
                qc = s.quantization_config;
                break;
            }
        }
        if (!qc) return null;

        const method = String(qc.quant_method || "").toLowerCase();
        const notes = [];
        const base = {
            method,
            excludePatterns: Array.isArray(qc.modules_to_not_convert)
                ? qc.modules_to_not_convert
                : (Array.isArray(qc.ignore) ? qc.ignore : []),
            targets: [],
            kvCacheScheme: qc.kv_cache_scheme || null,
            activations: null
        };

        if (method === "mxfp4") {
            // gpt-oss style: MoE experts in MXFP4, rest bf16. Trained
            // quantization-aware — MXFP4 IS the native checkpoint.
            return Object.assign(base, {
                family: "mxfp4", label: "mxfp4", qFormat: "mxfp4",
                qBytesPerParam: QFORMAT.mxfp4.bytes,
                groupSize: 32, bits: 4, modeled: true, notes
            });
        }

        if (method === "fp8") {
            // DeepSeek style: FP8 weights + separate higher-precision scales.
            return Object.assign(base, {
                family: "fp8",
                label: "fp8" + (qc.fmt ? " (" + qc.fmt + ")" : ""),
                qFormat: "fp8", qBytesPerParam: QFORMAT.fp8.bytes,
                groupSize: 0, bits: 8, modeled: true, notes
            });
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
            const g0 = groups.find((g) => g && g.weights) || {};
            base.targets = Array.isArray(g0.targets) ? g0.targets : [];
            base.activations = g0.input_activations || null;
            const label = "compressed-tensors" +
                (qc.format ? " (" + qc.format + ")" : "");
            if (!specs.length) {
                notes.push("compressed-tensors config without a weight spec; " +
                           "set the quantized split manually.");
                return Object.assign(base, {
                    family: "compressed-tensors", label, qFormat: null,
                    qBytesPerParam: null, groupSize: 0, bits: 0,
                    modeled: false, notes
                });
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
            // Group-scale width: MX formats declare torch.uint8 (E8M0);
            // compressed-tensors otherwise defaults to a 16-bit scale.
            const scaleDtype = String(w.scale_dtype || "").toLowerCase();
            const scaleBytes = /uint8|e8m0|float8|fp8/.test(scaleDtype) ? 1 : 2;

            if (type === "float" && bits === 4) {
                const isNv = gs === 16;
                const fmt  = isNv ? "nvfp4" : "mxfp4";
                return Object.assign(base, {
                    family: fmt, label, qFormat: fmt,
                    qBytesPerParam: weightSpecBytes(4, gs || 32,
                        /uint8|e8m0/.test(scaleDtype) || !scaleDtype ? 1 : scaleBytes),
                    groupSize: gs || 32, bits: 4, modeled: true, notes
                });
            }
            if (type === "float" && bits === 8) {
                // Group-scaled FP8 = MXFP8 (scales ride along); otherwise
                // plain FP8 with separate scale tensors.
                return gs > 0
                    ? Object.assign(base, {
                        family: "mxfp8", label, qFormat: "mxfp8",
                        qBytesPerParam: weightSpecBytes(8, gs, 1),
                        groupSize: gs, bits: 8, modeled: true, notes })
                    : Object.assign(base, {
                        family: "fp8", label, qFormat: "fp8",
                        qBytesPerParam: 1, groupSize: 0, bits: 8,
                        modeled: true, notes });
            }
            if (type === "int" && bits === 8) {
                return Object.assign(base, {
                    family: "int8", label, qFormat: "int8",
                    qBytesPerParam: 1, groupSize: gs, bits: 8,
                    modeled: true, notes
                });
            }
            if (type === "int" && bits === 4) {
                // INT4 with group scales (Kimi-K2.5 "pack-quantized"):
                // 0.5 B payload + scaleBytes per group element.
                const b = weightSpecBytes(4, gs, scaleBytes);
                return Object.assign(base, {
                    family: "int4", label,
                    qFormat: gs > 0 ? "q4s" : "int4",
                    qBytesPerParam: b, groupSize: gs, bits: 4,
                    modeled: true, notes
                });
            }
            notes.push("Unrecognised compressed-tensors weight spec (" +
                       type + "/" + bits + "-bit); set the split manually.");
            return Object.assign(base, {
                family: "compressed-tensors", label, qFormat: null,
                qBytesPerParam: null, groupSize: 0, bits: bits || 0,
                modeled: false, notes
            });
        }

        // Generic fallback for any future quant_method: transformers
        // enforces the quant_method discriminator + (usually) a bits field
        // and a modules_to_not_convert list, so surface what we can and
        // let the histogram / deep-inspect tiers refine it.
        const bits = qc.bits || qc.num_bits ||
            (qc.load_in_4bit ? 4 : (qc.load_in_8bit ? 8 : 0));
        const gs = typeof qc.group_size === "number" ? qc.group_size : 0;
        if (bits === 4 || bits === 8) {
            notes.push("Quantization method \"" + (method || "unknown") +
                       "\" is not fully modeled; assuming " + bits +
                       "-bit payload" + (gs ? " (group size " + gs + ")" : "") +
                       " — verify with Deep inspect or set the split manually.");
            return Object.assign(base, {
                family: method || "unknown",
                label: (method || "unknown") + " (generic)",
                qFormat: bits === 4 ? (gs ? "q4s" : "int4") : "int8",
                qBytesPerParam: weightSpecBytes(bits, gs, 2),
                groupSize: gs, bits, modeled: false, notes
            });
        }
        notes.push("Quantization method \"" + (method || "unknown") +
                   "\" detected but not modeled; set 'Quantized params' " +
                   "and format manually.");
        return Object.assign(base, {
            family: method || "unknown", label: method || "unknown",
            qFormat: null, qBytesPerParam: null, groupSize: 0, bits: 0,
            modeled: false, notes
        });
    }

    // ============================================================
    // HF API histogram interpretation (Tier 1 + Tier 3 cross-check)
    // ============================================================
    /** Sum the root-checkpoint *.safetensors sibling sizes (bytes).
     *  Excludes secondary copies (original/, metal/, gguf/, ...). */
    function sumCheckpointBytes(siblings) {
        if (!Array.isArray(siblings)) return 0;
        let sum = 0, matched = 0;
        for (const s of siblings) {
            if (!s || typeof s.rfilename !== "string") continue;
            if (!/^model[^/]*\.safetensors$/.test(s.rfilename)) continue;
            const size = typeof s.size === "number"
                ? s.size
                : (s.lfs && typeof s.lfs.size === "number" ? s.lfs.size : 0);
            if (size > 0) { sum += size; matched++; }
        }
        return matched > 0 ? sum : 0;
    }

    /** Which histogram buckets hold the quantized payload for a family. */
    function quantBucketKeys(family) {
        switch (family) {
            case "mxfp4":
            case "nvfp4": return ["U8", "F4"];
            case "int4":  return ["U8", "I32", "I8"];
            case "fp8":   return ["F8_E4M3", "F8_E5M2"];
            case "mxfp8": return ["F8_E4M3", "F8_E5M2", "U8"];
            case "int8":  return ["I8", "U8"];
            default:      return [];
        }
    }

    /**
     * Split the HF API per-dtype parameter counts into logical quantized /
     * non-quantized params.
     *
     * The Hub has served BOTH of these conventions for the same repo:
     *   "logical": counts are logical parameters, group scales excluded
     *              (current convention — verified to reconcile with disk
     *              bytes on gpt-oss-120b/20b, Kimi-K3, Kimi-K2.5).
     *   "packed":  counts are stored elements — packed U8 bytes counted as
     *              2 params each plus scale bytes as 1 each (older payloads).
     * When `diskBytes` is known we score both interpretations against the
     * actual checkpoint size and keep the closer one; otherwise "logical".
     *
     * Returns { qParams, nqParams, nqBytes, convention, predictedDiskBytes,
     *           diskDeltaPct, notes[] }
     */
    function interpretDtypeMap(quant, dtypeMap, diskBytes) {
        const notes = [];
        const qKeys = quantBucketKeys(quant.family);
        const gs = quant.groupSize || 32;
        const qb = quant.qBytesPerParam != null
            ? quant.qBytesPerParam
            : qformatBytes(quant.qFormat, 1);

        let qRaw = 0, nqParams = 0, nqBytes = 0;
        for (const [k, v] of Object.entries(dtypeMap)) {
            if (typeof v !== "number" || !isFinite(v) || v <= 0) continue;
            const key = k.toUpperCase();
            const isF8 = key === "F8_E4M3" || key === "F8_E5M2";
            if (qKeys.indexOf(key) !== -1) {
                // NVFP4 keeps its FP8 group scales in the F8 bucket; they are
                // already folded into qBytesPerParam — don't double-count.
                if (isF8 && quant.family === "nvfp4") continue;
                qRaw += v;
            } else {
                nqParams += v;
                nqBytes  += v * dtypeBytes(key);
            }
        }

        // Candidate interpretations of the quant bucket count.
        const candidates = [];
        candidates.push({ convention: "logical", qParams: qRaw });
        if (quant.bits === 4) {
            // packed U8/I32 elements → logical params, minus ride-along scales.
            const unit = dtypeMap.I32 ? 8 : 2; // int4-in-I32 packs 8, in-U8 packs 2
            if (quant.family === "mxfp4" && !dtypeMap.I32) {
                // Old convention counted blocks as 2 logical per byte AND the
                // per-group E8M0 scale bytes as params: logical = raw·gs/(gs+1).
                candidates.push({ convention: "packed+scales",
                                  qParams: qRaw * gs / (gs + 1) });
            } else {
                candidates.push({ convention: "packed",
                                  qParams: qRaw * unit });
            }
        }

        let best = candidates[0];
        if (diskBytes > 0) {
            let bestDelta = Infinity;
            for (const c of candidates) {
                const predicted = c.qParams * qb + nqBytes;
                const delta = Math.abs(predicted - diskBytes) / diskBytes;
                if (delta < bestDelta) { bestDelta = delta; best = c; }
            }
        }

        const predictedDiskBytes = best.qParams * qb + nqBytes;
        const diskDeltaPct = diskBytes > 0
            ? Math.abs(predictedDiskBytes - diskBytes) / diskBytes * 100
            : null;
        if (diskDeltaPct != null && diskDeltaPct > 5) {
            notes.push("Derived weight bytes disagree with the checkpoint " +
                       "size on disk by " + diskDeltaPct.toFixed(1) +
                       "% — treat the quantized split as approximate " +
                       "(use Deep inspect for exact numbers).");
        }
        if (best.convention !== "logical") {
            notes.push("HF param histogram appears to use packed-element " +
                       "counts (older Hub convention); corrected to logical " +
                       "parameters using the checkpoint size.");
        }
        return {
            qParams: best.qParams, nqParams, nqBytes,
            convention: best.convention,
            predictedDiskBytes, diskDeltaPct, notes
        };
    }

    // ============================================================
    // Linear-attention / recurrent state geometry (config-derived)
    // ============================================================
    /**
     * Per-layer, per-sequence fixed state for linear/recurrent layers,
     * derived from HF config fields (never model names):
     *   - Kimi KDA (`linear_attn_config`): S = heads·d_k·d_v plus a short
     *     conv state over the q/k/v streams.
     *   - Qwen3-Next gated DeltaNet (`linear_*` fields): S = v_heads·d_k·d_v
     *     plus conv over (2·key_dim + value_dim).
     *   - Mamba-style (`state_size`/`mamba_d_state`): inner·d_state plus
     *     conv_kernel·inner.
     * State is held at the compute dtype (assumed 2 B) — engines vary
     * (some keep FP32 SSM state), so this is a floor, flagged as derived.
     *
     * Returns { bytesPerLayer, source } — bytesPerLayer 0 when unknown.
     */
    function deriveLinearStateBytes(cfg) {
        const B = 2; // assume bf16 state
        if (cfg && cfg.linear_attn_config &&
            typeof cfg.linear_attn_config === "object" &&
            cfg.linear_attn_config.num_heads && cfg.linear_attn_config.head_dim) {
            const la = cfg.linear_attn_config;
            const heads = clampInt(la.num_heads, 1, 1 << 16, 1);
            const hd    = clampInt(la.head_dim, 1, 1 << 16, 128);
            const k     = clampInt(la.short_conv_kernel_size, 0, 1 << 10, 0);
            const state = heads * hd * hd * B;
            const conv  = k > 0 ? 3 * heads * hd * k * B : 0;
            return { bytesPerLayer: state + conv, source: "kda (linear_attn_config)" };
        }
        if (cfg && cfg.linear_key_head_dim && cfg.linear_value_head_dim &&
            cfg.linear_num_value_heads) {
            const dk = clampInt(cfg.linear_key_head_dim, 1, 1 << 16, 128);
            const dv = clampInt(cfg.linear_value_head_dim, 1, 1 << 16, 128);
            const vh = clampInt(cfg.linear_num_value_heads, 1, 1 << 16, 1);
            const kh = clampInt(cfg.linear_num_key_heads, 1, 1 << 16, vh);
            const k  = clampInt(cfg.linear_conv_kernel_dim, 0, 1 << 10, 0);
            const state = vh * dk * dv * B;
            const conv  = k > 0 ? (2 * kh * dk + vh * dv) * k * B : 0;
            return { bytesPerLayer: state + conv, source: "gated-deltanet (linear_* fields)" };
        }
        const dState = cfg && (cfg.mamba_d_state || cfg.state_size);
        if (dState) {
            const ds = clampInt(dState, 1, 1 << 20, 16);
            const hidden = clampInt(cfg.hidden_size, 1, 1 << 24, 4096);
            const expand = cfg.mamba_expand || cfg.expand || 2;
            const inner  = clampInt(cfg.mamba_d_ssm || cfg.intermediate_size ||
                                    hidden * expand, 1, 1 << 26, hidden * 2);
            const k = clampInt(cfg.mamba_d_conv || cfg.conv_kernel, 0, 1 << 10, 4);
            const state = inner * ds * B;
            const conv  = k > 0 ? inner * k * B : 0;
            return { bytesPerLayer: state + conv, source: "mamba (state_size fields)" };
        }
        return { bytesPerLayer: 0, source: null };
    }

    // ============================================================
    // Sizing math (pure — consumed by the UI's readState() output)
    // ============================================================
    /**
     * Weight bytes, disk vs resident.
     * s: { params, qParams, wBytes, qBytes } in billions / bytes-per-param.
     * meta (optional): { diskBytes, nativeQBytes } — checkpoint-native info
     * captured at preset/HF load time. Resident follows the form's selected
     * quantized format (which is how "dequantize to BF16" is expressed);
     * disk always follows the checkpoint-native layout.
     */
    function computeWeightBytes(s, meta) {
        const qP = Math.min(s.qParams || 0, s.params || 0);
        const nqP = (s.params || 0) - qP;
        const resident = nqP * 1e9 * s.wBytes + qP * 1e9 * (s.qBytes || 0);
        let disk = resident;
        let diskSource = "derived (same as resident)";
        if (meta && meta.diskBytes > 0) {
            disk = meta.diskBytes;
            diskSource = "checkpoint file sizes";
        } else if (meta && meta.nativeQBytes > 0 && qP > 0) {
            disk = nqP * 1e9 * s.wBytes + qP * 1e9 * meta.nativeQBytes;
            diskSource = "derived (native format)";
        }
        return { residentBytes: resident, diskBytes: disk, diskSource, qP, nqP };
    }

    /** KV bytes/token growth model per attention type. */
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

    /**
     * Sequence-growing KV total (paged) + fixed linear-layer state.
     * The recurrent/conv state of linear-attention layers is per-sequence
     * and seq-length-independent: state = linearLayers · stateBytes · batch.
     * It is not paged (engines allocate it densely), so it is added after
     * the paged-efficiency division.
     */
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
        const paged = raw / Math.max(0.01, s.pagedEff);
        const state = (s.linearLayers || 0) * (s.stateBytes || 0) *
                      Math.max(1, s.batch || 1);
        return { kvBytes: paged + state, pagedKvBytes: paged, stateBytes: state };
    }

    // ============================================================
    // HF config.json → baseline mapping
    // ============================================================
    /**
     * Map a Hugging Face `config.json` (+ /api/models metadata) into the
     * calculator's baseline shape. Pure: no DOM, no fetch.
     * Returns { baseline, info, warnings }.
     */
    function buildHfBaseline(repoId, raw, meta) {
        const found = findLmConfig(raw);
        const cfg = found.cfg;

        const layers  = clampInt(cfg.num_hidden_layers,  1, 4096, 1);
        const qHeads  = clampInt(cfg.num_attention_heads, 1, 4096, 1);

        // KV head count: HF's official key is `num_key_value_heads`, but
        // Falcon (and a few older models) use `num_kv_heads`. Default to
        // qHeads (i.e. MHA) when neither is reported.
        const rawKv = cfg.num_key_value_heads != null
            ? cfg.num_key_value_heads
            : (cfg.num_kv_heads != null ? cfg.num_kv_heads : qHeads);
        const kvHeads = clampInt(rawKv, 1, 4096, qHeads);

        // Head dim: prefer explicit head_dim. For DeepSeek-style MLA configs,
        // use qk_nope_head_dim + qk_rope_head_dim if those are set. Otherwise
        // fall back to hidden_size / qHeads.
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
            architectures: cfg.architectures || raw.architectures || [],
            nestedKey: found.nestedKey
        };

        // Detect attention type.
        let attn = "mha";
        const baseline = {
            "kvcc-layers":  layers,
            "kvcc-qheads":  qHeads,
            "kvcc-kvheads": kvHeads,
            "kvcc-headdim": headDim,
            // Always stamp so switching models can't leave stale state.
            "kvcc-linear-layers": 0,
            "kvcc-state-bytes": 0
        };

        const layerTypes = Array.isArray(cfg.layer_types) ? cfg.layer_types : null;
        const linearTypeRe = /linear|mamba|gated_delta|recurrent|kda|conv/;

        if (cfg.kv_lora_rank != null) {
            // DeepSeek MLA family.
            attn = "mla";
            baseline["kvcc-mla-lora"] = clampInt(cfg.kv_lora_rank, 1, 65536, 512);
            baseline["kvcc-mla-rope"] = clampInt(
                cfg.qk_rope_head_dim != null ? cfg.qk_rope_head_dim : 64,
                0, 65536, 64
            );
            // Hybrid MLA + linear attention (Kimi K3 / Kimi-Linear): only
            // the full-attention layers keep a sequence-growing MLA KV
            // cache; the linear-attention layers hold O(1) state.
            const fal = cfg.linear_attn_config &&
                        cfg.linear_attn_config.full_attn_layers;
            let fullCount = 0;
            if (Array.isArray(fal) && fal.length > 0 && fal.length < layers) {
                fullCount = fal.length;
            } else if (layerTypes &&
                       layerTypes.some((t) => linearTypeRe.test(t))) {
                fullCount = layerTypes.filter(
                    (t) => !linearTypeRe.test(t)).length;
            }
            if (fullCount > 0 && fullCount < layers) {
                baseline["kvcc-layers"] = fullCount;
                info.attnLayers  = fullCount;
                info.totalLayers = layers;
                const linearLayers = layers - fullCount;
                const st = deriveLinearStateBytes(cfg);
                baseline["kvcc-linear-layers"] = linearLayers;
                baseline["kvcc-state-bytes"]   = Math.round(st.bytesPerLayer / 1024);
                info.stateSource = st.source;
                info.stateBytesPerLayer = st.bytesPerLayer;
                warnings.push("Hybrid MLA + linear attention: " + fullCount +
                              " of " + layers + " layers use full (MLA) " +
                              "attention; the " + linearLayers +
                              " linear-attention layers hold O(1) state" +
                              (st.bytesPerLayer
                                  ? " (~" + Math.round(st.bytesPerLayer / 1024) +
                                    " KiB/layer/sequence, " + st.source + ")"
                                  : " (size unknown — set it under Advanced)") +
                              ". Layers set to " + fullCount + ".");
            }
        } else if (
            cfg.linear_attn_config ||
            cfg.linear_layer_indices ||
            cfg.linear_num_value_heads != null ||
            cfg.full_attention_interval != null ||
            cfg.attn_layer_indices != null ||
            (layerTypes && layerTypes.some((t) => linearTypeRe.test(t)))
        ) {
            // Hybrid linear-state + attention (Qwen3-Next, Jamba, ...).
            attn = "hybrid";
            let attnLayers = 0;
            if (Array.isArray(cfg.attn_layer_indices)) {
                attnLayers = cfg.attn_layer_indices.length;
            } else if (cfg.linear_attn_config &&
                       Array.isArray(cfg.linear_attn_config.full_attn_layers)) {
                attnLayers = cfg.linear_attn_config.full_attn_layers.length;
            } else if (layerTypes) {
                attnLayers = layerTypes.filter(
                    (t) => /^(full_|sliding_)?attention$/i.test(t)
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
                const period = cfg.linear_attn_config.linear_attn_period;
                if (period >= 2) attnLayers = Math.max(1, Math.floor(layers / period));
            }
            if (!attnLayers) {
                attnLayers = Math.max(1, Math.round(layers / 4));
                warnings.push("Hybrid model: couldn't determine attention-layer " +
                              "count from config; assuming 1-in-4.");
            }
            baseline["kvcc-hybrid-attn"] = attnLayers;
            const linearLayers = Math.max(0, layers - attnLayers);
            const st = deriveLinearStateBytes(cfg);
            baseline["kvcc-linear-layers"] = linearLayers;
            baseline["kvcc-state-bytes"]   = Math.round(st.bytesPerLayer / 1024);
            info.stateSource = st.source;
            info.stateBytesPerLayer = st.bytesPerLayer;
            if (linearLayers > 0 && !st.bytesPerLayer) {
                warnings.push("Linear/recurrent layers detected but their " +
                              "per-layer state size couldn't be derived from " +
                              "the config; state counted as 0 — set " +
                              "'State per linear layer' under Advanced.");
            }
        } else if (cfg.sliding_window && cfg.use_sliding_window !== false) {
            // Gemma-3-style local/global SWA, or pure SWA.
            const sw = clampInt(cfg.sliding_window, 1, 1 << 24, 1024);
            const pattern = clampInt(cfg.sliding_window_pattern, 0, 1024, 0);
            attn = "swa";
            baseline["kvcc-swa-window"] = sw;

            // Exact local/global counts from layer_types when it fully
            // describes the stack (gpt-oss: 18 sliding + 18 full).
            let slidingCount = 0, fullCount = 0;
            if (layerTypes) {
                slidingCount = layerTypes.filter((t) => /sliding/i.test(t)).length;
                fullCount = layerTypes.filter((t) => /^full_attention$/i.test(t)).length;
            }
            if (slidingCount > 0 &&
                layerTypes && slidingCount + fullCount === layerTypes.length) {
                baseline["kvcc-swa-local"]  = slidingCount;
                baseline["kvcc-swa-global"] = fullCount;
            } else if (pattern && pattern > 1) {
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

        // Quantization: read quantization_config (top-level or nested).
        const quant = detectQuantization(raw);
        if (quant) {
            for (const n of quant.notes) warnings.push(n);
            info.quant = quant;
        }

        // Weight precision of the non-quantized share. Transformers renamed
        // `torch_dtype` → `dtype` (Kimi-K2.5 ships only the new key).
        const dtype = (cfg.torch_dtype || raw.torch_dtype ||
                       cfg.dtype || raw.dtype || "").toLowerCase();
        if (dtype === "bfloat16" || dtype === "float16" || dtype === "fp16" ||
            dtype === "bf16" || dtype === "float32") {
            // (FP32 checkpoints are near-universally served in BF16/FP16;
            // the precision select has no 4-byte option by design.)
            baseline["kvcc-wprec"] = 2;
        } else if (dtype === "float8_e4m3fn" || dtype === "fp8" || dtype === "int8") {
            baseline["kvcc-wprec"] = 1;
        } else if (dtype === "int4") {
            baseline["kvcc-wprec"] = 0.5;
        } else if (quant && quant.modeled) {
            // Quantized repo without a dtype (gpt-oss): the un-quantized
            // modules (attention, embeddings, ...) are bf16.
            baseline["kvcc-wprec"] = 2;
        }

        // KV-cache scheme (compressed-tensors): when a repo declares its
        // KV cache quantized, suggest the matching KV precision.
        if (quant && quant.kvCacheScheme &&
            typeof quant.kvCacheScheme === "object") {
            const bits = quant.kvCacheScheme.num_bits;
            if (bits === 8) baseline["kvcc-kvprec"] = 1;
            else if (bits === 4) baseline["kvcc-kvprec"] = 0.5;
            if (bits === 8 || bits === 4) {
                warnings.push("Repo declares a quantized KV-cache scheme (" +
                              bits + "-bit); KV precision set to match.");
                info.kvCacheScheme = quant.kvCacheScheme;
            }
        }

        // Disk bytes of the shipped checkpoint (Tier 3).
        const diskBytes = meta ? sumCheckpointBytes(meta.siblings) : 0;
        if (diskBytes > 0) {
            info.diskBytes = diskBytes;
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
            const split = interpretDtypeMap(quant, dtypeMap, diskBytes);
            for (const n of split.notes) warnings.push(n);
            const totalB = (split.qParams + split.nqParams) / 1e9;
            const qB     = split.qParams / 1e9;
            baseline["kvcc-params"]  = Number(totalB.toFixed(2));
            baseline["kvcc-qparams"] = Number(qB.toFixed(2));
            baseline["kvcc-qformat"] = quant.qFormat || "mxfp4";
            info.paramSource = "safetensors.parameters (dtype split, " +
                               split.convention + ")";
            info.params  = totalB;
            info.qParams = qB;
            info.dtypeMap = dtypeMap;
            info.split = split;
        } else if (st && typeof st.total === "number") {
            const billions = st.total / 1e9;
            baseline["kvcc-params"] = Number(billions.toFixed(2));
            info.paramSource = "safetensors.total";
            info.params = billions;
            if (dtypeMap) info.dtypeMap = dtypeMap;
            if (quant && quant.modeled) {
                warnings.push("Quantized repo but the API returned no " +
                              "per-dtype parameter map; param count may be " +
                              "inflated and the quantized split was not set " +
                              "— adjust 'Quantized params' manually.");
            }
        } else {
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

    // ============================================================
    // Tier 4: safetensors header classification (Deep inspect)
    // ============================================================
    // Packed-tensor name suffixes → logical params per stored element.
    const PACKED_SUFFIXES = [
        { re: /_blocks$/,        unit: 2,   kind: "packed 4-bit (2/byte)" },
        { re: /weight_packed$/,  unit: 8,   kind: "packed int4 (8/int32)" },
        { re: /\bqweight$/,      unit: 8,   kind: "GPTQ/AWQ qweight (8/int32)" },
        { re: /_scales$|weight_scale(_inv)?$/, unit: 0, kind: "group scales" },
        { re: /_bias$|\.bias$/,  unit: 1,   kind: "bias" }
    ];

    /**
     * Aggregate one shard's safetensors header JSON.
     * header: { tensorName: { dtype, shape[], data_offsets:[b,e] }, ... }
     * Accumulates into `acc` (create with makeInspectAccumulator()).
     */
    function accumulateHeader(acc, header, quant) {
        for (const [name, t] of Object.entries(header)) {
            if (name === "__metadata__" || !t || !Array.isArray(t.shape)) continue;
            const dt = String(t.dtype || "").toUpperCase();
            const elems = t.shape.reduce((a, b) => a * b, 1) || 0;
            const bytes = Array.isArray(t.data_offsets)
                ? Math.max(0, t.data_offsets[1] - t.data_offsets[0])
                : elems * dtypeBytes(dt);

            acc.tensorCount++;
            acc.totalBytes += bytes;
            if (!acc.byDtype[dt]) acc.byDtype[dt] = { elems: 0, bytes: 0, tensors: 0 };
            acc.byDtype[dt].elems += elems;
            acc.byDtype[dt].bytes += bytes;
            acc.byDtype[dt].tensors++;

            // Logical-parameter attribution.
            let unit = 1, isScale = false;
            for (const p of PACKED_SUFFIXES) {
                if (p.re.test(name)) {
                    if (p.unit === 0) { isScale = true; }
                    else { unit = p.unit; }
                    break;
                }
            }
            const excluded = quant && quant.excludePatterns &&
                             quant.excludePatterns.length
                ? moduleMatches(name, quant.excludePatterns)
                : false;
            const isQuantDtype = dt === "U8" || dt === "I32" || dt === "I8" ||
                                 dt === "F8_E4M3" || dt === "F8_E5M2" || dt === "F4";
            if (isScale) {
                acc.scaleBytes += bytes;
            } else if (quant && !excluded && isQuantDtype) {
                // Logical params per stored element: explicit packed-name
                // suffixes win; otherwise infer from dtype + quant bit width.
                let per = unit;
                if (per === 1) {
                    if (dt === "U8" && quant.bits === 4) per = 2;
                    else if (dt === "I32" && quant.bits === 4) per = 8;
                    else if (dt === "I32" && quant.bits === 8) per = 4;
                }
                acc.qParams += elems * per;
                acc.qBytes += bytes;
            } else {
                acc.nqParams += elems;
                acc.nqBytes  += bytes;
            }
        }
        return acc;
    }

    function makeInspectAccumulator() {
        return { tensorCount: 0, totalBytes: 0, byDtype: {},
                 qParams: 0, qBytes: 0, nqParams: 0, nqBytes: 0,
                 scaleBytes: 0 };
    }

    // ============================================================
    // Exports
    // ============================================================
    const KVCCModel = {
        QFORMAT, DTYPE_BYTES,
        qformatBytes, weightSpecBytes, dtypeBytes,
        clampInt, moduleMatches, findLmConfig,
        detectQuantization, sumCheckpointBytes, quantBucketKeys,
        interpretDtypeMap, deriveLinearStateBytes,
        computeWeightBytes, computeKvBytesPerToken, computeTotalKvBytes,
        buildHfBaseline,
        accumulateHeader, makeInspectAccumulator
    };

    root.KVCCModel = KVCCModel;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = KVCCModel;
    }
})(typeof globalThis !== "undefined" ? globalThis : this);
