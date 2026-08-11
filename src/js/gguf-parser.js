const GGUF_MAGIC = 0x46554747; // "GGUF"
const GGUF_TYPES = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0',
  9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K',
  15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S',
  20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_M', 23: 'IQ4_XS', 24: 'I8', 25: 'I16',
  26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16',
  // Metadata value types (these are NOT tensor types, they're the KV types)
};

// GGUF metadata value types
const GGUF_META_TYPE = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3,
  UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7,
  STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

class Reader {
  constructor(buffer, byteOffset = 0) {
    // Accept both ArrayBuffer and ArrayBufferView (Uint8Array/Node Buffer)
    if (buffer instanceof ArrayBuffer) {
      this.dv = new DataView(buffer, byteOffset);
    } else {
      // ArrayBufferView (e.g. Uint8Array from fetch, or Node Buffer from readFileSync)
      this.dv = new DataView(buffer.buffer, buffer.byteOffset + byteOffset, buffer.byteLength - byteOffset);
    }
    this.offset = 0;
  }

  remaining() { return this.dv.byteLength - this.offset; }

  /**
   * Try to skip `count` elements of `elemType` without reading their values.
   * For string elements we must read each length (variable-width), so this
   * walks them. For fixed-size scalars it computes the byte size directly.
   * Returns true if all elements fit in the remaining buffer, false if we
   * hit the end (partial range fetch).
   */
  trySkip(count, elemType) {
    const FIXED_SIZE = {
      [GGUF_META_TYPE.UINT8]: 1, [GGUF_META_TYPE.INT8]: 1,
      [GGUF_META_TYPE.UINT16]: 2, [GGUF_META_TYPE.INT16]: 2,
      [GGUF_META_TYPE.UINT32]: 4, [GGUF_META_TYPE.INT32]: 4,
      [GGUF_META_TYPE.FLOAT32]: 4, [GGUF_META_TYPE.BOOL]: 1,
      [GGUF_META_TYPE.UINT64]: 8, [GGUF_META_TYPE.INT64]: 8,
      [GGUF_META_TYPE.FLOAT64]: 8,
    };
    if (elemType in FIXED_SIZE) {
      const bytes = count * FIXED_SIZE[elemType];
      if (this.offset + bytes > this.dv.byteLength) return false;
      this.offset += bytes;
      return true;
    }
    // String elements: must read each length to skip
    if (elemType === GGUF_META_TYPE.STRING) {
      for (let i = 0; i < count; i++) {
        if (this.remaining() < 8) return false;
        const len = this.u64();
        if (this.remaining() < len) return false;
        this.offset += len;
      }
      return true;
    }
    return false;
  }

  u8()  { const v = this.dv.getUint8(this.offset); this.offset += 1; return v; }
  i8()  { const v = this.dv.getInt8(this.offset); this.offset += 1; return v; }
  u16() { const v = this.dv.getUint16(this.offset, true); this.offset += 2; return v; }
  i16() { const v = this.dv.getInt16(this.offset, true); this.offset += 2; return v; }
  u32() { const v = this.dv.getUint32(this.offset, true); this.offset += 4; return v; }
  i32() { const v = this.dv.getInt32(this.offset, true); this.offset += 4; return v; }
  u64() {
    // Read as two u32, combine (JS numbers are f64, safe up to 2^53)
    const lo = this.dv.getUint32(this.offset, true);
    const hi = this.dv.getUint32(this.offset + 4, true);
    this.offset += 8;
    return lo + hi * 0x100000000;
  }
  i64() {
    const lo = this.dv.getUint32(this.offset, true);
    const hi = this.dv.getInt32(this.offset + 4, true);
    this.offset += 8;
    return lo + hi * 0x100000000;
  }
  f32() { const v = this.dv.getFloat32(this.offset, true); this.offset += 4; return v; }
  f64() { const v = this.dv.getFloat64(this.offset, true); this.offset += 8; return v; }
  bool() { return this.u8() !== 0; }

  string() {
    const len = this.u64();
    const bytes = new Uint8Array(this.dv.buffer, this.dv.byteOffset + this.offset, len);
    this.offset += len;
    return new TextDecoder('utf-8').decode(bytes);
  }

  scalar(type) {
    switch (type) {
      case GGUF_META_TYPE.UINT8:   return this.u8();
      case GGUF_META_TYPE.INT8:    return this.i8();
      case GGUF_META_TYPE.UINT16:  return this.u16();
      case GGUF_META_TYPE.INT16:   return this.i16();
      case GGUF_META_TYPE.UINT32:  return this.u32();
      case GGUF_META_TYPE.INT32:   return this.i32();
      case GGUF_META_TYPE.FLOAT32: return this.f32();
      case GGUF_META_TYPE.BOOL:    return this.bool();
      case GGUF_META_TYPE.STRING:  return this.string();
      case GGUF_META_TYPE.UINT64:  return this.u64();
      case GGUF_META_TYPE.INT64:   return this.i64();
      case GGUF_META_TYPE.FLOAT64: return this.f64();
      default: throw new Error(`Unknown GGUF scalar type ${type} at offset ${this.offset}`);
    }
  }

  array(elemType, preReadLen) {
    const len = preReadLen ?? this.u64();
    const arr = [];
    for (let i = 0; i < len; i++) {
      arr.push(this.scalar(elemType));
    }
    return arr;
  }
}

/**
 * Parse a GGUF header from an ArrayBuffer (full file or partial range).
 *
 * @param {ArrayBuffer} buffer - the GGUF bytes (at least the metadata block)
 * @param {object} [opts] - { maxMetadata: 200 } truncate long metadata values
 * @returns {{magic:number, version:number, nTensors:number, metadata:object, tensorInfos:Array}}
 * @throws if the buffer is too small or the magic doesn't match
 */
export function parseGgufHeader(buffer, opts = {}) {
  const maxMetaLen = opts.maxMetadata ?? 500;
  const r = new Reader(buffer);

  const magic = r.u32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Not a GGUF file: magic 0x${magic.toString(16)} != 0x${GGUF_MAGIC.toString(16)}`);
  }

  const version = r.u32();
  let nTensors, nKV;
  if (version >= 3) {
    nTensors = r.u64();
    nKV = r.u64();
  } else {
    nTensors = r.u32();
    nKV = r.u32();
  }

  const metadata = {};
  for (let i = 0; i < nKV; i++) {
    // On a partial range fetch we may hit the end of the buffer before reading
    // all KV pairs (e.g. tokenizer token arrays are huge). Bail out gracefully —
    // architecture metadata always comes first in practice.
    if (r.remaining() < 16) break;

    const key = r.string();
    const valType = r.u32();

    let value;
    if (valType === GGUF_META_TYPE.ARRAY) {
      const elemType = r.u32();
      // Skip large arrays (tokenizer tokens), but keep small architecture arrays
      const arrayLen = r.u64();
      if (arrayLen > 1000) {
        // Try to skip; if we run out of buffer, stop (partial range fetch).
        const skipOk = r.trySkip(arrayLen, elemType);
        value = skipOk ? `<${arrayLen} items>` : `<${arrayLen} items, truncated>`;
        if (!skipOk) break;
      } else {
        value = r.array(elemType, arrayLen);
      }
    } else {
      value = r.scalar(valType);
      if (typeof value === 'string' && value.length > maxMetaLen) {
        value = `<${value.length} chars>`;
      }
    }
    metadata[key] = value;
  }

  // Parse tensor info (name, n_dims, dims, type, data offset)
  // We don't need the tensor data, but the info is right after metadata.
  // If buffer is too small, we just skip tensors.
  const tensorInfos = [];
  for (let i = 0; i < nTensors; i++) {
    if (r.remaining() < 8) break;
    const name = r.string();
    const nDims = r.u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(r.u64());
    const ttype = r.u32();
    const dataOffset = r.u64();
    tensorInfos.push({ name, dims, type: GGUF_TYPES[ttype] || `type_${ttype}`, ttype, dataOffset });
  }

  return { magic, version, nTensors, nKV, metadata, tensorInfos };
}

/**
 * Extract a DFlash draft architecture object (compatible with vram-estimator's
 * calcKVCache) from parsed GGUF metadata.
 *
 * DFlash GGUF keys (prefix "dflash."):
 *   dflash.block_count             = number of layers
 *   dflash.attention.head_count_kv = KV heads
 *   dflash.attention.key_length    = head_dim
 *   dflash.attention.sliding_window = SWA window size (0 if none)
 *   dflash.attention.sliding_window_pattern = bool array, True = SWA layer
 *
 * @param {object} metadata - from parseGgufHeader()
 * @returns {object|null} arch object, or null if not a dflash model
 */
export function extractDflashArch(metadata) {
  if (metadata['general.architecture'] !== 'dflash') return null;

  const numLayers = metadata['dflash.block_count'] || 0;
  const numKVHeads = metadata['dflash.attention.head_count_kv'] || 0;
  const headDim = metadata['dflash.attention.key_length'] || 0;
  const swa = metadata['dflash.attention.sliding_window'] || 0;

  // Build layerTypes from the sliding_window_pattern (all-true means all SWA)
  const swaPattern = metadata['dflash.attention.sliding_window_pattern'];
  let layerTypes = null;
  if (Array.isArray(swaPattern) && swaPattern.length === numLayers) {
    layerTypes = swaPattern.map(isSwa => isSwa ? 'sliding_attention' : 'full_attention');
  }

  return {
    hiddenSize: metadata['dflash.embedding_length'] || 0,
    numLayers,
    numAttnHeads: metadata['dflash.attention.head_count'] || 0,
    numKVHeads,
    headDim,
    numGlobalKVHeads: null,
    globalHeadDim: null,
    slidingWindow: swa,
    layerTypes,
    fullAttnInterval: 1,
    isHybrid: false,
    mtpLayers: 0,
    vocabSize: 202048,
  };
}

/**
 * Extract architecture params from a Gemma4 assistant (MTP sidecar) GGUF header.
 *
 * Gemma4-assistant GGUF keys (prefix "gemma4-assistant."):
 *   general.architecture                          = "gemma4-assistant"
 *   gemma4-assistant.block_count                  = number of MTP layers (e.g. 4)
 *   gemma4-assistant.embedding_length             = draft hidden size (e.g. 1024)
 *   gemma4-assistant.embedding_length_out         = target hidden size (e.g. 2816)
 *   gemma4-assistant.attention.shared_kv_layers   = layers sharing KV (== block_count
 *                                                   for Gemma4 — ALL layers share the
 *                                                   target's KV cache, zero draft KV)
 * No vocab key: the assistant borrows the target model's vocab.
 *
 * VRAM cost is therefore just the sidecar weights + a draft compute buffer (the
 * draft context runs llama_decode against the borrowed KV — see llama-model.cpp:2362
 * share callback + speculative.cpp:1489).
 *
 * @param {object} metadata - from parseGgufHeader()
 * @returns {object|null} { numLayers, hiddenSize, vocabSize, nEmbdOut, sharedKvLayers } or null
 */
export function extractMtpArch(metadata) {
  if (metadata['general.architecture'] !== 'gemma4-assistant') return null;

  const numLayers = metadata['gemma4-assistant.block_count'] || 0;
  const sharedKvLayers = metadata['gemma4-assistant.attention.shared_kv_layers'] || 0;

  return {
    numLayers,
    hiddenSize: metadata['gemma4-assistant.embedding_length'] || 0,
    nEmbdOut: metadata['gemma4-assistant.embedding_length_out'] || 0,
    sharedKvLayers,
    // No vocab key in the sidecar header — inherits the target's vocab.
    // Use the Gemma4 default as a fallback for the compute buffer estimate.
    vocabSize: 262144,
  };
}

/**
 * Extract CLIP vision encoder params from an mmproj GGUF header.
 *
 * CLIP GGUF keys (clip-impl.h):
 *   clip.vision.embedding_length  = n_embd
 *   clip.vision.block_count       = n_layer
 *   clip.vision.feed_forward_length = n_ff
 *   clip.vision.image_size        = image_size (e.g. 1024)
 *   clip.vision.patch_size        = patch_size (e.g. 14)
 *
 * @param {object} metadata - from parseGgufHeader()
 * @returns {object|null} {nEmbmd, nLayer, nFf, imageSize, patchSize, nPatches} or null
 */
export function extractClipArch(metadata) {
  const nEmbmd = metadata['clip.vision.embedding_length'];
  if (!nEmbmd) return null;

  const nLayer = metadata['clip.vision.block_count'] || 0;
  const nFf = metadata['clip.vision.feed_forward_length'] || 0;
  const imageSize = metadata['clip.vision.image_size'] || 0;
  const patchSize = metadata['clip.vision.patch_size'] || 0;
  const nPatches = (imageSize && patchSize)
    ? Math.floor(imageSize / patchSize) ** 2
    : 0;

  return { nEmbmd, nLayer, nFf, imageSize, patchSize, nPatches };
}
