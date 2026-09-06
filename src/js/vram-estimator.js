// ---------------------------------------------------------------------------
// Quant ordering
// ---------------------------------------------------------------------------

export const QUANT_ORDER = [
  'F32','BF16','F16','Q8_K_XL','UD-Q8_K_XL','Q8_0','Q8_K_L','UD-Q8_K_L',
  'Q6_K_XL','UD-Q6_K_XL','Q6_K_L','UD-Q6_K_L','Q6_K_M','UD-Q6_K_M','Q6_K','UD-Q6_K',
  'Q5_K_XL','UD-Q5_K_XL','Q5_K_L','UD-Q5_K_L','Q5_K_M','UD-Q5_K_M','Q5_K_S','UD-Q5_K_S',
  'Q4_K_XL','UD-Q4_K_XL','Q4_1','Q4_K_M','UD-Q4_K_M','Q4_K_S','UD-Q4_K_S','IQ4_NL','UD-IQ4_NL','Q4_0','IQ4_XS','UD-IQ4_XS',
  'Q3_K_XL','UD-Q3_K_XL','Q3_K_M','UD-Q3_K_M','Q3_K_S','UD-Q3_K_S','IQ3_M','UD-IQ3_M','IQ3_XXS','UD-IQ3_XXS','IQ3_S','UD-IQ3_S',
  'Q2_K_XL','UD-Q2_K_XL','Q2_K','UD-Q2_K','IQ2_M','UD-IQ2_M','IQ2_XS','UD-IQ2_XS','IQ2_XXS','UD-IQ2_XXS','IQ1_M','UD-IQ1_M','IQ1_S','UD-IQ1_S',
];

export function sortQuants(quants) {
  return quants.sort((a, b) => {
    const ia = QUANT_ORDER.indexOf(a.quant);
    const ib = QUANT_ORDER.indexOf(b.quant);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });
}

/**
 * Extract the quant name from a GGUF filename.
 *   "Muse-Glimmer-30B-UD-Q4_K_XL.gguf" -> "UD-Q4_K_XL"
 *   "BF16/Muse-Glimmer-30B-BF16-00001-of-00002.gguf" -> "BF16"
 */
export function parseQuant(filename) {
  const base = filename.replace(/\.gguf$/i, '').split('/').pop().replace(/-\d{5}-of-\d{5}$/i, '');
  const match = base.match(/(-?(?:UD-)?(?:IQ\d+_[A-Z0-9_]+|Q\d+_[A-Z0-9_]+|BF\d+|F\d+))$/i);
  return match ? match[1].toUpperCase().replace(/^-/, '') : base.toUpperCase();
}

export function isFirstPart(filename) {
  return /-00001-of-\d{5}$/i.test(filename.replace(/\.gguf$/i, ''));
}

/**
 * Accept plain repo IDs ("org/model") or full HF URLs and return "org/model".
 *   "https://huggingface.co/unsloth/Foo-GGUF" -> "unsloth/Foo-GGUF"
 *   "unsloth/Foo-GGUF" -> "unsloth/Foo-GGUF"
 */
export function normalizeRepo(input) {
  let s = (input || '').trim();
  s = s.replace(/[?#].*$/, '');
  s = s.replace(/^https?:\/\//i, '');
  s = s.replace(/^(?:www\.)?(?:huggingface\.co|hf\.co)\//i, '');
  const parts = s.split('/').filter(Boolean);
  return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : s;
}

// ---------------------------------------------------------------------------
// KV cache / VRAM math
// ---------------------------------------------------------------------------

/**
 * CUDA MATRIX_ROW_PADDING (ggml-cuda.cu:906) adds only a SINGLE extra row of
 * padding to the whole per-layer tensor (so the last quantized row can't read
 * out of bounds), not a per-row inflation. For a cache with kv_size rows that
 * extra row is ~1/kv_size of the tensor — a rounding error. So no BPP overhead.
 * F16/F32 are unquantized and never padded regardless.
 */
export function effectiveBpp(rowElems, bpp) {
  return { bpp, pct: 0 };
}

/**
 * Extract architecture params from a transformers config.json.
 * Handles: standard dense, GQA, hybrid (linear_attention), MTP, sliding-window,
 * and Gemma4 dual-KV (num_global_key_value_heads / global_head_dim).
 */
export function extractArch(config) {
  const c = config.text_config || config;
  // MTP layers: transformers call them mtp_num_hidden_layers (Qwen) or
  // num_nextn_predict_layers (GLM-5.x NextN head).
  const mtpLayers = c.mtp_num_hidden_layers || c.num_nextn_predict_layers || 0;
  const numLayers = (c.num_hidden_layers || 0) + mtpLayers;
  const layerTypes = c.layer_types || null;
  // MLA (Multi-head Latent Attention) — GLM-5.x / DeepSeek-style. The KV cache
  // stores ONE compressed latent per layer (kv_lora_rank + rope dims) and has NO
  // separate V tensor (llama-kv-cache.cpp: has_v = !is_mla()). GLM-5.x also keeps a
  // sparse-attention indexer K cache on a subset of "full" layers
  // (llama-kv-cache-dsa.cpp: both caches use type_k, so the K quant applies to both).
  const isMla = !!c.kv_lora_rank;
  // KDA (Gated DeltaNet) recurrent layers — GLM-5.3-Flash "linear_attention"
  // layers. They keep a conv state + S matrix per layer (both F32), no KV cache.
  const nKdaLayers = Array.isArray(layerTypes)
    ? layerTypes.filter(t => t === 'linear_attention').length
    : 0;
  // MLA latent cache only exists on the full-attention (DSA) layers. For
  // GLM-5.3 (glm-dsa) every trunk layer is MLA; for GLM-5.3-Flash (glm5next)
  // only the deepseek_sparse_attention layers are (34 of 45 are KDA).
  const nMlaLayers = isMla ? (Array.isArray(layerTypes)
    ? layerTypes.filter(t => t === 'deepseek_sparse_attention').length || (c.num_hidden_layers || 0)
    : (c.num_hidden_layers || 0)) : 0;
  // Indexer K cache: only layers that are BOTH "full" indexers AND full-attention
  // layers own one (glm5next filter_idx = trunk && !is_recr). GLM-5.3 has no
  // layer_types, so count the "full" entries directly (21 of 78).
  let indexerLayers = 0;
  if (Array.isArray(c.indexer_types)) {
    if (Array.isArray(layerTypes)) {
      indexerLayers = c.indexer_types.filter((t, i) =>
        (typeof t === 'string' ? t === 'full' : !!t) && layerTypes[i] === 'deepseek_sparse_attention'
      ).length;
    } else {
      indexerLayers = c.indexer_types.filter(t => (typeof t === 'string' ? t === 'full' : !!t)).length;
    }
  }
  // Gemma4: num_kv_shared_layers layers at the end share their KV cache with
  // earlier layers and don't allocate their own. Only the first
  // (numLayers - num_kv_shared_layers) layers have KV (llama-hparams.cpp:267).
  const numKvSharedLayers = c.num_kv_shared_layers || 0;
  return {
    hiddenSize: c.hidden_size,
    numLayers,
    numAttnHeads: c.num_attention_heads,
    numKVHeads: c.num_key_value_heads || c.num_attention_heads,
    headDim: c.head_dim || (c.hidden_size / c.num_attention_heads),
    // Gemma 4: full-attention layers use global KV heads + head dim
    numGlobalKVHeads: c.num_global_key_value_heads || null,
    globalHeadDim: c.global_head_dim || null,
    // MLA (GLM-5.x / DeepSeek-style): compressed-latent K cache, no V tensor.
    isMla,
    mlaKLatent: isMla ? (c.kv_lora_rank + (c.qk_rope_head_dim || 0)) : 0,
    nMlaLayers,
    indexerKeyLen: c.index_head_dim || 0,
    indexerLayers,
    // The pooling indexer (glm5next, kpool>0) caches key+gate+pooled-key (3 heads)
    // and is forced to F16 (gate feeds a softmax — llama-model.cpp).
    indexerHeads: c.index_kpool ? 3 : 1,
    indexerForcedF16: !!c.index_kpool,
    // KDA recurrent state (GLM-5.3-Flash "linear_attention" layers):
    // conv state 3*(d_conv-1)*heads*head_dim + S state heads*head_dim^2, both F32.
    // (llama-hparams.cpp n_embd_r/n_embd_s KDA branch; n_head = num_attention_heads)
    kdaHeadDim: c.linear_attn_config?.head_dim || 0,
    kdaDConv: c.linear_attn_config?.short_conv_kernel_size || 0,
    nKdaLayers,
    slidingWindow: c.sliding_window || 0,
    layerTypes,
    fullAttnInterval: c.full_attention_interval || 1,
    numKvSharedLayers,
    ssmDConv: c.ssm_d_conv || c.linear_conv_kernel_dim,
    ssmDInner: c.ssm_d_inner || (c.linear_num_value_heads || 0) * (c.linear_value_head_dim || 0),
    ssmDState: c.ssm_d_state,
    ssmNGroup: c.ssm_n_group || c.linear_num_key_heads,
    isHybrid: !!(c.full_attention_interval && c.full_attention_interval > 1),
    mtpLayers,
    vocabSize: c.vocab_size || 128256,
  };
}

/**
 * KV cache: K + V split, each with own BPP.
 * Handles per-layer attention types (full vs sliding), and the Gemma4 dual-KV
 * case where full-attn layers use different head counts/dims.
 *
 * @param {number} ctx     - context length (n_ctx)
 * @param {number} ubatch  - physical batch size (n_ubatch)
 * @param {object} arch    - architecture params (from extractArch)
 * @param {number} kBpp    - K cache bits-per-element
 * @param {number} vBpp    - V cache bits-per-element
 * @returns {{kGB,vGB,sizeGB,nAttnLayers,nSWA,padPct}}
 */
export function calcKVCache(ctx, ubatch, arch, kBpp, vBpp) {
  const mainLayers = arch.numLayers - arch.mtpLayers;

  // MLA (GLM-5.x / DeepSeek-style): one compressed-latent K tensor per MLA layer,
  // NO V tensor (llama-kv-cache.cpp: has_v = !is_mla()). GLM-5.x additionally
  // keeps a sparse-attention indexer K cache on a subset of layers
  // (llama-kv-cache-dsa.cpp / llama-memory-hybrid.cpp). Both caches use type_k,
  // so the K quant applies to both (the pooling indexer in glm5next is forced
  // to F16 instead — gate feeds a softmax).
  if (arch.isMla) {
    const eKBpp = effectiveBpp(arch.mlaKLatent, kBpp);
    const idxBpp = arch.indexerForcedF16 ? 2.0 : eKBpp.bpp;
    const nIdx = Math.min(arch.indexerLayers, mainLayers);
    const kBytes = arch.mlaKLatent * ctx * arch.nMlaLayers * eKBpp.bpp;
    const idxBytes = arch.indexerKeyLen * arch.indexerHeads * ctx * nIdx * idxBpp;
    return {
      kGB: (kBytes + idxBytes) / 1e9, vGB: 0,
      sizeGB: (kBytes + idxBytes) / 1e9,
      nAttnLayers: arch.nMlaLayers, nSWA: 0, padPct: eKBpp.pct,
      kLatentGB: kBytes / 1e9, indexerGB: idxBytes / 1e9,
      nIndexerLayers: nIdx,
    };
  }

  // Gemma4: layers with index >= nKvLayers share their KV with earlier layers
  // and don't allocate their own cache tensors (llama-hparams.cpp:267 has_kv()).
  const nKvLayers = arch.numKvSharedLayers
    ? mainLayers - arch.numKvSharedLayers
    : mainLayers;
  let kBytes = 0, vBytes = 0;
  let nAttnCount = 0, nSWA = 0;
  let maxPct = 0;

  if (arch.layerTypes) {
    for (let il = 0; il < arch.layerTypes.length; il++) {
      if (il >= nKvLayers) break; // shared layers: no own KV cache
      const lt = arch.layerTypes[il];
      if (lt === 'full_attention') {
        const nKVH = arch.numGlobalKVHeads || arch.numKVHeads;
        const hDim = arch.globalHeadDim || arch.headDim;
        const nEmbK = nKVH * hDim;
        const eKBpp = effectiveBpp(nEmbK, kBpp);
        const eVBpp = effectiveBpp(nEmbK, vBpp);
        maxPct = Math.max(maxPct, eKBpp.pct, eVBpp.pct);
        kBytes += nEmbK * ctx * eKBpp.bpp;
        vBytes += nEmbK * ctx * eVBpp.bpp;
        nAttnCount++;
      } else if (lt === 'sliding_attention') {
        const nEmbK = arch.numKVHeads * arch.headDim;
        const eKBpp = effectiveBpp(nEmbK, kBpp);
        const eVBpp = effectiveBpp(nEmbK, vBpp);
        maxPct = Math.max(maxPct, eKBpp.pct, eVBpp.pct);
        const swaCtx = Math.min((arch.slidingWindow || ctx) + ubatch, ctx);
        kBytes += nEmbK * swaCtx * eKBpp.bpp;
        vBytes += nEmbK * swaCtx * eVBpp.bpp;
        nSWA++;
        nAttnCount++;
      }
    }
  } else {
    const nAttnLayers = Math.floor(mainLayers / arch.fullAttnInterval);
    const nEmbK = arch.numKVHeads * arch.headDim;
    const eKBpp = effectiveBpp(nEmbK, kBpp);
    const eVBpp = effectiveBpp(nEmbK, vBpp);
    maxPct = Math.max(maxPct, eKBpp.pct, eVBpp.pct);
    kBytes = nEmbK * ctx * nAttnLayers * eKBpp.bpp;
    vBytes = nEmbK * ctx * nAttnLayers * eVBpp.bpp;
    nAttnCount = nAttnLayers;
  }

  const totalBytes = kBytes + vBytes;
  return { kGB: kBytes / 1e9, vGB: vBytes / 1e9, sizeGB: totalBytes / 1e9, nAttnLayers: nAttnCount, nSWA, padPct: maxPct };
}

/**
 * Copy of `arch` with MTP layers excluded, for estimating a run WITHOUT
 * speculative drafting from the same file (`--spec-type` omitted → llama.cpp
 * load_mtp=false, qwen35.cpp:42 skips the MTP tensors entirely — no weights
 * loaded, no draft KV allocated). `numLayers` from a config-derived arch
 * already includes mtpLayers, so subtract them back out.
 */
export function withMtpDisabled(arch) {
  if (!arch || !arch.mtpLayers) return arch;
  return { ...arch, mtpLayers: 0, numLayers: arch.numLayers - arch.mtpLayers };
}

/**
 * Recurrent state for hybrid models. Two shapes:
 * - Mamba-style (Qwen3.6 linear_attention): conv + SSM state.
 * - KDA / Gated DeltaNet (GLM-5.3-Flash "linear_attention"): conv state for
 *   Q,K,V + the S matrix, both F32 (llama-hparams.cpp n_embd_r/n_embd_s KDA
 *   branch, llama-memory-recurrent.cpp). Per layer, per row:
 *     conv = 3 * (d_conv - 1) * n_head * head_dim
 *     S    = n_head * head_dim^2
 *   Rows = 1 + nRsSeq (same cell layout as the KV cache).
 */
export function calcRecurrentState(arch, nRsSeq) {
  // KDA / Gated DeltaNet (GLM-5.3-Flash "linear_attention" layers): keyed off
  // layer_types + linear_attn_config, independent of the Mamba isHybrid flag
  // (Flash has no full_attention_interval).
  if (arch.nKdaLayers > 0 && (arch.kdaHeadDim || 0) > 0) {
    const nAttnLayers = Array.isArray(arch.layerTypes)
      ? arch.layerTypes.filter(t => t === 'deepseek_sparse_attention').length
      : 0;
    const nRecurLayers = arch.nKdaLayers;
    const nHead = arch.numAttnHeads;
    const dConv = arch.kdaDConv || 4;
    const embdR = 3 * (dConv - 1) * nHead * arch.kdaHeadDim;
    const embdS = arch.kdaHeadDim * arch.kdaHeadDim * nHead;
    const nRows = 1 + nRsSeq;
    const bytesPerLayer = (embdR + embdS) * nRows * 4; // f32
    const totalBytes = nRecurLayers * bytesPerLayer;
    return { sizeGB: totalBytes / 1e9, nRecurLayers, nAttnLayers };
  }
  if (!arch.isHybrid || !arch.ssmDState) return { sizeGB: 0, nRecurLayers: 0 };
  const nMain = arch.numLayers - arch.mtpLayers;
  const nAttnLayers = Math.floor(nMain / arch.fullAttnInterval);
  const nRecurLayers = nMain - nAttnLayers;
  const embdR = (arch.ssmDConv - 1) * (arch.ssmDInner + 2 * arch.ssmNGroup * arch.ssmDState);
  const embdS = arch.ssmDState * arch.ssmDInner;
  const nRows = 1 + nRsSeq;
  const bytesPerLayer = (embdR + embdS) * nRows * 4;
  const totalBytes = nRecurLayers * bytesPerLayer;
  return { sizeGB: totalBytes / 1e9, nRecurLayers, nAttnLayers };
}

/**
 * MTP (multi-token prediction) draft KV cache.
 * The draft layers are full-attention layers (no SWA). For MLA models they are
 * dense MLA layers, so the draft cache is the same compressed latent as the
 * main cache, quantized with the draft K quant — no V tensor. GLM-5.3-Flash's
 * NextN block additionally reads the indexer (PR #27754: the MTP ctx filter
 * gives the nextn layer a lid cache, forced F16), so add one indexer cell.
 */
export function calcMtpKV(ctx, arch, mtpkBpp, mtpvBpp) {
  if (!arch.mtpLayers) return { sizeGB: 0, kGB: 0, vGB: 0, padPct: 0 };
  if (arch.isMla) {
    const eKBpp = effectiveBpp(arch.mlaKLatent, mtpkBpp);
    const idxBpp = arch.indexerForcedF16 ? 2.0 : eKBpp.bpp;
    const kBytes = ctx * arch.mtpLayers * arch.mlaKLatent * eKBpp.bpp;
    const idxBytes = arch.indexerForcedF16
      ? ctx * arch.mtpLayers * arch.indexerKeyLen * arch.indexerHeads * idxBpp
      : 0;
    return { sizeGB: (kBytes + idxBytes) / 1e9, kGB: (kBytes + idxBytes) / 1e9, vGB: 0, padPct: eKBpp.pct };
  }
  const nEmbK = arch.numKVHeads * arch.headDim;
  const eKBpp = effectiveBpp(nEmbK, mtpkBpp);
  const eVBpp = effectiveBpp(nEmbK, mtpvBpp);
  const kBytes = ctx * arch.mtpLayers * nEmbK * eKBpp.bpp;
  const vBytes = ctx * arch.mtpLayers * nEmbK * eVBpp.bpp;
  return { sizeGB: (kBytes + vBytes) / 1e9, kGB: kBytes / 1e9, vGB: vBytes / 1e9, padPct: Math.max(eKBpp.pct, eVBpp.pct) };
}

/**
 * Compute buffer estimate: logits tensor + base overhead.
 * Hybrid models (Qwen3.6) use a larger base buffer due to recurrent state ops.
 */
export function calcComputeBuffer(ubatch, arch) {
  const n_vocab = arch.vocabSize || 128256;
  const logitsBytes = n_vocab * ubatch * 2;
  const baseBytes = arch.isHybrid ? 0.8 * 1e9 : 0.5 * 1e9;
  return (logitsBytes + baseBytes) / 1e9;
}

/**
 * mmproj (CLIP vision encoder) compute buffer estimate.
 *
 * When mmproj runs on GPU, ggml_backend_sched_reserve() allocates activation
 * buffers for the vision encoder forward pass (clip.cpp:3492). The scheduler's
 * graph allocator (ggml_gallocr) performs lifetime analysis, so intermediate
 * activations are reused across layers. The peak memory is thus driven by a
 * single layer's simultaneously-live tensors:
 *   - FFN intermediate: [n_patches, n_ff] of f32 (dominant)
 *   - residual + LN + attn I/O: ~3 × [n_patches, n_embd] of f32
 *
 * @param {object|null} clipArch - from extractClipArch(), or null
 * @returns {number} compute buffer size in GB
 */
export function calcMmprojCompute(clipArch) {
  if (!clipArch || !clipArch.nPatches || !clipArch.nLayer) return 0;
  const { nPatches, nLayer, nEmbmd, nFf } = clipArch;
  // Peak per-layer activations (ggml reuses buffers across layers)
  const peakBytes = nPatches * (nFf + 3 * nEmbmd) * 4; // sizeof(f32) = 4
  // Small overhead per layer for persistent graph metadata
  const layerOverhead = nLayer * nEmbmd * 4;
  return (peakBytes + layerOverhead) / 1e9;
}

/**
 * MTP sidecar compute buffer estimate.
 *
 * The sidecar (gemma4-assistant) shares the target's KV cache, so it has zero
 * draft KV. The only extra VRAM beyond the sidecar weights is a draft compute
 * buffer: the draft context calls llama_decode (speculative.cpp:1489) and needs
 * a logits tensor [vocab × ubatch] plus a small base buffer, same shape as the
 * main model's compute buffer.
 *
 * @param {number} ubatch     - physical batch size (n_ubatch)
 * @param {object} mtpArch    - from extractMtpArch() { vocabSize, ... }
 * @returns {number} compute buffer size in GB
 */
export function calcMtpDraftCompute(ubatch, mtpArch) {
  if (!mtpArch) return 0;
  const n_vocab = mtpArch.vocabSize || 262144;
  const logitsBytes = n_vocab * ubatch * 2; // fp16 logits
  const baseBytes = 0.25 * 1e9; // small base (4 layers, lightweight)
  return (logitsBytes + baseBytes) / 1e9;
}

/**
 * Aggregate all per-quant VRAM totals for a given model config + user settings.
 *
 * @param {object} opts
 * @param {Array}  opts.ggufFiles      - [{path, size}, ...] (pre-filtered to real quants)
 * @param {number} opts.ctx            - context length
 * @param {number} opts.ubatch         - physical batch size
 * @param {object} opts.arch           - architecture params (from extractArch) or null
 * @param {number} opts.kBpp           - K cache BPP
 * @param {number} opts.vBpp           - V cache BPP
 * @param {number} opts.mtpkBpp        - MTP K cache BPP
 * @param {number} opts.mtpvBpp        - MTP V cache BPP
 * @param {object} [opts.dflash]       - DFlash sidecar {arch, sizeGB} or null
 * @param {number} [opts.dflashkBpp]   - DFlash K cache BPP (defaults to main kBpp)
 * @param {number} [opts.dflashvBpp]   - DFlash V cache BPP (defaults to main vBpp)
 * @param {object} [opts.mtpSidecar]   - MTP sidecar {arch, sizeGB} or null
 * @returns {{quants, kv, rs, mtp, dflash, mtpSidecar, computeBuf}}
 */
export function estimateVram(opts) {
  const { ggufFiles, ctx, ubatch, arch, kBpp, vBpp, mtpkBpp, mtpvBpp, dflash, mtpSidecar } = opts;
  const dflashkBpp = opts.dflashkBpp ?? kBpp;
  const dflashvBpp = opts.dflashvBpp ?? vBpp;

  // KV cache for main model
  const kv = arch
    ? calcKVCache(ctx, ubatch, arch, kBpp, vBpp)
    : { sizeGB: 0, kGB: 0, vGB: 0, nAttnLayers: 0, nSWA: 0, padPct: 0 };

  // Recurrent state (hybrid models)
  const nRsSeq = arch && arch.mtpLayers ? 2 : 0;
  const rs = arch ? calcRecurrentState(arch, nRsSeq) : { sizeGB: 0, nRecurLayers: 0 };

  // MTP draft KV
  const mtp = arch ? calcMtpKV(ctx, arch, mtpkBpp, mtpvBpp) : { sizeGB: 0, kGB: 0, vGB: 0 };

  // Compute buffer
  const computeBuf = arch ? calcComputeBuffer(ubatch, arch) : 1;

  // DFlash sidecar: weights (from file size) + draft KV
  let dflashTotal = 0;
  let dflashKV = null;
  if (dflash && dflash.arch) {
    dflashKV = calcKVCache(ctx, ubatch, dflash.arch, dflashkBpp, dflashvBpp);
    dflashTotal = dflash.sizeGB + dflashKV.sizeGB;
  } else if (dflash && dflash.sizeGB) {
    dflashTotal = dflash.sizeGB;
  }

  // MTP sidecar: weights + draft compute buffer (zero KV — shared with target)
  const mtpSidecarCompute = mtpSidecar && mtpSidecar.arch
    ? calcMtpDraftCompute(ubatch, mtpSidecar.arch)
    : 0;
  const mtpSidecarTotal = mtpSidecar ? (mtpSidecar.sizeGB + mtpSidecarCompute) : 0;

  // Per-quant totals
  const quantMap = {};
  for (const f of ggufFiles) {
    const quant = parseQuant(f.path);
    const sizeGB = f.size / 1e9;
    if (!quantMap[quant]) quantMap[quant] = { quant, sizeGB: 0, file: f.path };
    quantMap[quant].sizeGB += sizeGB;
    if (isFirstPart(f.path)) quantMap[quant].file = f.path;
  }

  const quants = sortQuants(Object.values(quantMap)).map(q => ({
    ...q,
    total: q.sizeGB + kv.sizeGB + rs.sizeGB + mtp.sizeGB + dflashTotal + mtpSidecarTotal + computeBuf,
  }));

  return {
    quants, kv, rs, mtp, computeBuf,
    dflash: { ...dflash, kv: dflashKV, total: dflashTotal },
    mtpSidecar: mtpSidecar
      ? { ...mtpSidecar, computeGB: mtpSidecarCompute, total: mtpSidecarTotal }
      : null,
  };
}
