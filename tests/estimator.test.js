import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  QUANT_ORDER, sortQuants, parseQuant, isFirstPart, normalizeRepo,
  effectiveBpp, extractArch, calcKVCache, calcMtpKV, calcComputeBuffer,
  calcRecurrentState, calcMmprojCompute, calcMtpDraftCompute,
  estimateVram, withMtpDisabled,
} from '../src/js/vram-estimator.js';

// ---------------------------------------------------------------------------
// Quant ordering & parsing
// ---------------------------------------------------------------------------

test('parseQuant extracts common quants', () => {
  assert.equal(parseQuant('Muse-Glimmer-30B-UD-Q4_K_XL.gguf'), 'UD-Q4_K_XL');
  assert.equal(parseQuant('Muse-Glimmer-30B-UD-IQ2_XS.gguf'), 'UD-IQ2_XS');
  assert.equal(parseQuant('Qwen3.6-27B-Q4_K_M.gguf'), 'Q4_K_M');
  assert.equal(parseQuant('model-Q8_0.gguf'), 'Q8_0');
  // Multi-part split files
  assert.equal(parseQuant('BF16/Muse-Glimmer-30B-BF16-00001-of-00002.gguf'), 'BF16');
});

test('parseQuant handles missing quants that previously broke sort', () => {
  // These are the exact quants from unsloth/Muse-Glimmer-30B-GGUF
  const files = [
    'BF16/Muse-Glimmer-30B-BF16-00001-of-00002.gguf',
    'Muse-Glimmer-30B-UD-IQ2_M.gguf',
    'Muse-Glimmer-30B-UD-IQ2_XS.gguf',
    'Muse-Glimmer-30B-UD-IQ2_XXS.gguf',
    'Muse-Glimmer-30B-UD-IQ3_M.gguf',
    'Muse-Glimmer-30B-UD-IQ3_XXS.gguf',
    'Muse-Glimmer-30B-UD-Q2_K_XL.gguf',
    'Muse-Glimmer-30B-UD-Q3_K_XL.gguf',
    'Muse-Glimmer-30B-UD-Q4_K_XL.gguf',
    'Muse-Glimmer-30B-UD-Q5_K_L.gguf',
    'Muse-Glimmer-30B-UD-Q5_K_M.gguf',
    'Muse-Glimmer-30B-UD-Q5_K_XL.gguf',
    'Muse-Glimmer-30B-UD-Q6_K_XL.gguf',
    'Muse-Glimmer-30B-UD-Q8_K_XL.gguf',
  ];
  const quants = files.map(f => ({ quant: parseQuant(f), file: f }));
  // All should resolve to a known entry in QUANT_ORDER (index != -1)
  for (const q of quants) {
    assert.notEqual(QUANT_ORDER.indexOf(q.quant), -1, `${q.quant} missing from QUANT_ORDER`);
  }

  sortQuants(quants);
  // First should be BF16, last should be the smallest quant
  assert.equal(quants[0].quant, 'BF16');
  // IQ2_XXS should be near the bottom
  assert.equal(quants[quants.length - 1].quant, 'UD-IQ2_XXS');
});

test('sortQuants sorts high to low quality', () => {
  const quants = [
    { quant: 'Q4_K_M' }, { quant: 'Q8_0' }, { quant: 'Q2_K' }, { quant: 'Q6_K' },
  ];
  sortQuants(quants);
  assert.deepEqual(quants.map(q => q.quant), ['Q8_0', 'Q6_K', 'Q4_K_M', 'Q2_K']);
});

test('isFirstPart detects split file part 1', () => {
  assert.equal(isFirstPart('model-Q4_K_M-00001-of-00003.gguf'), true);
  assert.equal(isFirstPart('model-Q4_K_M-00002-of-00003.gguf'), false);
});

test('normalizeRepo handles URLs and bare IDs', () => {
  assert.equal(normalizeRepo('unsloth/Foo-GGUF'), 'unsloth/Foo-GGUF');
  assert.equal(normalizeRepo('https://huggingface.co/unsloth/Foo-GGUF'), 'unsloth/Foo-GGUF');
  assert.equal(normalizeRepo('https://huggingface.co/unsloth/Foo-GGUF/tree/main'), 'unsloth/Foo-GGUF');
  assert.equal(normalizeRepo('hf.co/unsloth/Foo-GGUF'), 'unsloth/Foo-GGUF');
});

// ---------------------------------------------------------------------------
// effectiveBpp (CUDA padding fix)
// ---------------------------------------------------------------------------

test('effectiveBpp returns no overhead (single guard row, not per-row)', () => {
  // 256 is not a multiple of 512, but the overhead should still be 0
  // because CUDA adds ONE guard row to the whole tensor, not per-row inflation
  const result = effectiveBpp(256, 1.0625);
  assert.equal(result.bpp, 1.0625);
  assert.equal(result.pct, 0);
});

// ---------------------------------------------------------------------------
// KV cache: Muse-Glimmer-30B (validated against llama.cpp)
// ---------------------------------------------------------------------------

const GLIMMER_ARCH = {
  numLayers: 52, mtpLayers: 0,
  numKVHeads: 2, headDim: 128,
  numGlobalKVHeads: null, globalHeadDim: null,
  slidingWindow: 2048,
  layerTypes: (() => {
    const types = [];
    for (let i = 0; i < 52; i++) types.push(i % 4 === 3 ? 'full_attention' : 'sliding_attention');
    return types;
  })(),
  fullAttnInterval: 1, isHybrid: false, vocabSize: 202048,
};

test('Muse-Glimmer-30B KV cache at q8_0 full ctx matches validated value', () => {
  const ctx = 131072, ubatch = 512, bpp = 1.0625;
  const kv = calcKVCache(ctx, ubatch, GLIMMER_ARCH, bpp, bpp);
  // 13 full-attn layers, 39 SWA layers
  assert.equal(kv.nAttnLayers, 52);
  assert.equal(kv.nSWA, 39);
  // Validated value: 0.98 GB (no CUDA padding inflation)
  assert.ok(Math.abs(kv.sizeGB - 0.98) < 0.01, `KV was ${kv.sizeGB}, expected ~0.98`);
  assert.equal(kv.padPct, 0); // no CUDA padding overhead
});

// ---------------------------------------------------------------------------
// KV cache: Qwen3.6-27B (validated against llama.cpp)
// ---------------------------------------------------------------------------

const QWEN36_CONFIG = {
  text_config: {
    num_hidden_layers: 64,
    mtp_num_hidden_layers: 1,
    num_attention_heads: 24,
    num_key_value_heads: 4,
    head_dim: 256,
    full_attention_interval: 4,
    layer_types: (() => {
      const types = [];
      for (let i = 0; i < 64; i++) types.push(i % 4 === 3 ? 'full_attention' : 'linear_attention');
      return types;
    })(),
    vocab_size: 248320,
    linear_conv_kernel_dim: 4,
    linear_num_key_heads: 16,
    linear_value_head_dim: 128,
    ssm_d_state: 128,
    linear_num_value_heads: 48,
    ssm_n_group: 16,
  },
};

test('Qwen3.6-27B KV cache at q8_0 full ctx matches validated value', () => {
  const arch = extractArch(QWEN36_CONFIG);
  const ctx = 131072, ubatch = 512, bpp = 1.0625;
  const kv = calcKVCache(ctx, ubatch, arch, bpp, bpp);
  // 16 full-attn layers (64/4), nEmbK = 4 * 256 = 1024
  assert.equal(kv.nAttnLayers, 16);
  assert.equal(kv.nSWA, 0);
  // Validated value: ~4.56 GB for KV, plus ~0.29 GB MTP
  assert.ok(Math.abs(kv.sizeGB - 4.56) < 0.02, `KV was ${kv.sizeGB}, expected ~4.56`);
});

test('Qwen3.6-27B MTP draft KV', () => {
  const arch = extractArch(QWEN36_CONFIG);
  const mtp = calcMtpKV(131072, arch, 1.0625, 1.0625);
  assert.ok(mtp.sizeGB > 0.2 && mtp.sizeGB < 0.4, `MTP was ${mtp.sizeGB}, expected ~0.29`);
});

// ---------------------------------------------------------------------------
// withMtpDisabled: estimate a run WITHOUT speculative drafting from an
// MTP-containing GGUF (llama.cpp skips the MTP tensors when spec decoding
// is off — no draft KV, no extra rs_seq)
// ---------------------------------------------------------------------------

test('withMtpDisabled strips MTP layers from the arch', () => {
  const arch = extractArch(QWEN36_CONFIG);
  assert.equal(arch.numLayers, 65); // 64 trunk + 1 MTP
  assert.equal(arch.mtpLayers, 1);

  const noMtp = withMtpDisabled(arch);
  assert.equal(noMtp.numLayers, 64);
  assert.equal(noMtp.mtpLayers, 0);
  // Original untouched
  assert.equal(arch.numLayers, 65);
  assert.equal(arch.mtpLayers, 1);
});

test('withMtpDisabled drops draft KV and recurrent-state penalty', () => {
  const arch = extractArch(QWEN36_CONFIG);
  const noMtp = withMtpDisabled(arch);
  const ctx = 131072, bpp = 1.0625;

  assert.equal(calcMtpKV(ctx, noMtp, bpp, bpp).sizeGB, 0);

  // Main KV is unchanged: trunk layers keep their cache either way
  assert.ok(Math.abs(calcKVCache(ctx, 512, arch, bpp, bpp).sizeGB
                  - calcKVCache(ctx, 512, noMtp, bpp, bpp).sizeGB) < 1e-9);

  // Recurrent state: nRsSeq=2 only when MTP drafting is active
  const rsWith = estimateVram({ ggufFiles: [{ path: 'm-Q4_K_M.gguf', size: 16e9 }], ctx, ubatch: 512, arch, kBpp: bpp, vBpp: bpp, mtpkBpp: bpp, mtpvBpp: bpp });
  const rsWithout = estimateVram({ ggufFiles: [{ path: 'm-Q4_K_M.gguf', size: 16e9 }], ctx, ubatch: 512, arch: noMtp, kBpp: bpp, vBpp: bpp, mtpkBpp: bpp, mtpvBpp: bpp });
  assert.ok(rsWith.mtp.sizeGB > 0, 'MTP KV should be counted when enabled');
  assert.equal(rsWithout.mtp.sizeGB, 0, 'MTP KV should vanish when disabled');
  assert.ok(rsWith.rs.sizeGB > rsWithout.rs.sizeGB, `rs_seq penalty should drop (${rsWith.rs.sizeGB} -> ${rsWithout.rs.sizeGB})`);
});

test('withMtpDisabled is a no-op for MTP-less arches', () => {
  const arch = extractArch(QWEN36_CONFIG);
  const noMtp = withMtpDisabled(arch);
  assert.equal(withMtpDisabled(noMtp), noMtp); // same reference, not a copy
  assert.equal(withMtpDisabled(null), null);
});

// ---------------------------------------------------------------------------
// KV cache: Gemma4-31B (dual KV heads)
// ---------------------------------------------------------------------------

const GEMMA4_CONFIG = {
  text_config: {
    num_hidden_layers: 60,
    num_attention_heads: 32,
    num_key_value_heads: 16,
    num_global_key_value_heads: 4,
    head_dim: 256,
    global_head_dim: 512,
    sliding_window: 1024,
    layer_types: (() => {
      const types = [];
      for (let i = 0; i < 60; i++) types.push((i + 1) % 6 === 0 ? 'full_attention' : 'sliding_attention');
      return types;
    })(),
    vocab_size: 262144,
  },
};

test('Gemma4-31B KV cache uses global KV heads for full-attn layers', () => {
  const arch = extractArch(GEMMA4_CONFIG);
  const ctx = 131072, ubatch = 512, bpp = 1.0625;
  const kv = calcKVCache(ctx, ubatch, arch, bpp, bpp);
  // 10 full-attn layers with global KV (4 * 512 = 2048), 50 SWA layers (16 * 256 = 4096 @ 1024 ctx)
  assert.equal(kv.nAttnLayers, 60);
  assert.equal(kv.nSWA, 50);
  // Validated value: ~6.37 GB
  assert.ok(Math.abs(kv.sizeGB - 6.37) < 0.05, `KV was ${kv.sizeGB}, expected ~6.37`);
});

// ---------------------------------------------------------------------------
// Gemma4 E4B: num_kv_shared_layers (layers without their own KV cache)
// ---------------------------------------------------------------------------

const GEMMA4_E4B_CONFIG = {
  text_config: {
    num_hidden_layers: 42,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    num_global_key_value_heads: null,
    head_dim: 256,
    global_head_dim: 512,
    sliding_window: 512,
    num_kv_shared_layers: 18,
    layer_types: (() => {
      const types = [];
      for (let i = 0; i < 42; i++) types.push((i + 1) % 6 === 0 ? 'full_attention' : 'sliding_attention');
      return types;
    })(),
    vocab_size: 262144,
  },
};

test('Gemma4 E4B respects num_kv_shared_layers (only 24/42 layers have KV)', () => {
  const arch = extractArch(GEMMA4_E4B_CONFIG);
  assert.equal(arch.numKvSharedLayers, 18);
  const ctx = 131072, ubatch = 512, bpp = 1.0625;
  const kv = calcKVCache(ctx, ubatch, arch, bpp, bpp);
  // Only first 24 layers count (42 - 18 shared)
  // Of those: 4 full-attn (indices 5,11,17,23) + 20 SWA
  assert.equal(kv.nAttnLayers, 24);
  assert.equal(kv.nSWA, 20);
  // Validated: ~1.09 GB, NOT ~1.92 GB (which would be all 42 layers)
  assert.ok(kv.sizeGB < 1.2, `KV was ${kv.sizeGB}, expected <1.2 (shared layers not counted)`);
  assert.ok(kv.sizeGB > 1.0, `KV was ${kv.sizeGB}, expected >1.0`);
});

// ---------------------------------------------------------------------------
// KV cache: GLM-5.3 (glm-dsa, MLA + sparse indexer)
// ---------------------------------------------------------------------------
// Ground truth from the real GGUF header (unsloth/GLM-5.3-GGUF Q8_0):
//   78 trunk layers + 1 NextN (nextn_predict_layers=1)
//   attention.key_length = 576 (kv_lora_rank 512 + rope 64), head_count_kv = 1
//   indexer: key_length 128, 21 of 78 layers are "full" indexers (glm-dsa.cpp default)

const GLM53_CONFIG = {
  hidden_size: 6144,
  num_hidden_layers: 78,
  num_attention_heads: 64,
  num_key_value_heads: 64,
  head_dim: 192,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  v_head_dim: 256,
  index_head_dim: 128,
  index_topk: 2048,
  num_nextn_predict_layers: 1,
  indexer_types: (() => {
    // glm-dsa.cpp GLM_5_2_DEFAULT_INDEXER_TYPES: 21 full (0,1,2, then every 4th)
    const types = new Array(78).fill('shared');
    [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74]
      .forEach(i => { types[i] = 'full'; });
    return types;
  })(),
  vocab_size: 154880,
};

test('GLM-5.3 extractArch reads MLA params', () => {
  const arch = extractArch(GLM53_CONFIG);
  assert.equal(arch.isMla, true);
  assert.equal(arch.mlaKLatent, 576); // 512 + 64
  assert.equal(arch.numLayers, 79);   // 78 trunk + 1 NextN
  assert.equal(arch.mtpLayers, 1);
  assert.equal(arch.indexerLayers, 21);
  assert.equal(arch.indexerKeyLen, 128);
  assert.equal(arch.indexerForcedF16, false);
});

test('GLM-5.3 KV cache at q8_0 full ctx: MLA latent + indexer, no V', () => {
  const arch = extractArch(GLM53_CONFIG);
  const ctx = 131072, bpp = 1.0625;
  const kv = calcKVCache(ctx, 512, arch, bpp, bpp);
  // K latent: 78 × 576 × 131072 × 1.0625 bytes (bpp is bytes/element here)
  const latentBytes = 78 * 576 * ctx * bpp;
  const idxBytes = 21 * 128 * ctx * bpp;
  assert.equal(kv.vGB, 0, 'MLA has no V tensor');
  assert.ok(Math.abs(kv.kLatentGB - latentBytes / 1e9) < 1e-9, `latent ${kv.kLatentGB}`);
  assert.ok(Math.abs(kv.indexerGB - idxBytes / 1e9) < 1e-9, `indexer ${kv.indexerGB}`);
  assert.ok(Math.abs(kv.sizeGB - (latentBytes + idxBytes) / 1e9) < 1e-9);
  // ~6.257 GB latent + ~0.374 GB indexer at 131K ctx, q8_0
  assert.ok(Math.abs(kv.sizeGB - 6.63) < 0.01, `KV was ${kv.sizeGB}, expected ~6.63`);
});

test('GLM-5.3 MTP draft KV is dense MLA (latent, no V, no indexer)', () => {
  const arch = extractArch(GLM53_CONFIG);
  const ctx = 131072, bpp = 0.5625; // q4_0 draft
  const mtp = calcMtpKV(ctx, arch, bpp, 0);
  const expected = 1 * 576 * ctx * bpp;
  assert.equal(mtp.vGB, 0);
  assert.ok(Math.abs(mtp.sizeGB - expected / 1e9) < 1e-9, `MTP was ${mtp.sizeGB}`);
  // ~0.0425 GB at 131K ctx q4_0
  assert.ok(Math.abs(mtp.sizeGB - 0.0425) < 0.001);
});

test('withMtpDisabled drops GLM-5.3 NextN layer and draft KV', () => {
  const arch = extractArch(GLM53_CONFIG);
  const noMtp = withMtpDisabled(arch);
  assert.equal(noMtp.numLayers, 78);
  const ctx = 131072, bpp = 1.0625;
  // Trunk KV unchanged, draft KV gone
  assert.ok(Math.abs(calcKVCache(ctx, 512, arch, bpp, bpp).sizeGB
                  - calcKVCache(ctx, 512, noMtp, bpp, bpp).sizeGB) < 1e-9);
  assert.equal(calcMtpKV(ctx, noMtp, bpp, 0).sizeGB, 0);
});

// ---------------------------------------------------------------------------
// KV cache + recurrent state: GLM-5.3-Flash (glm5next, KDA + DSA-MLA)
// ---------------------------------------------------------------------------
// Ground truth from the real GGUF header (unsloth/GLM-5.3-Flash-GGUF Q8_0) +
// zai-org/GLM-5.3-Flash config.json (PR #27754 source):
//   block_count 46 = 45 trunk + 1 NextN
//   layer_types: 34 linear_attention (KDA) + 11 deepseek_sparse_attention (DSA)
//   MLA: kv_lora_rank 512, qk_rope_head_dim 0 -> latent 512, no V
//   indexer: 11 DSA layers x 3 heads (kpool=4) x 128, F16 forced
//   KDA: head_dim 128, conv_kernel 4, n_head 64 -> F32 recurrent state

const GLM53F_CONFIG = {
  hidden_size: 4096,
  num_hidden_layers: 45,
  num_attention_heads: 64,
  num_key_value_heads: 64,
  kv_lora_rank: 512,
  qk_nope_head_dim: 256,
  qk_rope_head_dim: 0,
  v_head_dim: 256,
  index_head_dim: 128,
  index_kpool: 4,
  index_topk: 2048,
  num_nextn_predict_layers: 1,
  layer_types: (() => {
    // DSA at 3,7,11,...,43 (every 4th starting at 3); rest are KDA
    const types = new Array(45).fill('linear_attention');
    for (let i = 3; i < 45; i += 4) types[i] = 'deepseek_sparse_attention';
    return types;
  })(),
  indexer_types: new Array(45).fill('full'),
  linear_attn_config: {
    num_heads: 64,
    head_dim: 128,
    short_conv_kernel_size: 4,
  },
  vocab_size: 154880,
};

test('GLM-5.3-Flash extractArch reads hybrid MLA params', () => {
  const arch = extractArch(GLM53F_CONFIG);
  assert.equal(arch.isMla, true);
  assert.equal(arch.mlaKLatent, 512);        // 512 + 0 (nope-only)
  assert.equal(arch.numLayers, 46);          // 45 trunk + 1 NextN
  assert.equal(arch.mtpLayers, 1);
  assert.equal(arch.nMlaLayers, 11);         // DSA layers only
  assert.equal(arch.nKdaLayers, 34);
  assert.equal(arch.indexerLayers, 11);      // full AND DSA
  assert.equal(arch.indexerKeyLen, 128);
  assert.equal(arch.indexerHeads, 3);        // kpool pooling indexer
  assert.equal(arch.indexerForcedF16, true);
  assert.equal(arch.kdaHeadDim, 128);
  assert.equal(arch.kdaDConv, 4);
});

test('GLM-5.3-Flash KV at 131K q8_0: 11 DSA latent + 11 DSA indexer (f16), no V', () => {
  const arch = extractArch(GLM53F_CONFIG);
  const ctx = 131072, bpp = 1.0625;
  const kv = calcKVCache(ctx, 512, arch, bpp, bpp);
  // latent: 11 x 512 x 131072 x 1.0625 = 0.77516 GB
  const latentBytes = 11 * 512 * ctx * bpp;
  // indexer: 11 x 3 x 128 x 131072 x 2.0 (F16) = 1.09951 GB
  const idxBytes = 11 * 3 * 128 * ctx * 2.0;
  assert.equal(kv.vGB, 0, 'MLA has no V tensor');
  assert.ok(Math.abs(kv.kLatentGB - latentBytes / 1e9) < 1e-9, `latent ${kv.kLatentGB}`);
  assert.ok(Math.abs(kv.indexerGB - idxBytes / 1e9) < 1e-9, `indexer ${kv.indexerGB}`);
  assert.ok(Math.abs(kv.sizeGB - (latentBytes + idxBytes) / 1e9) < 1e-9);
  // latent 0.7844 + indexer 1.1073 = ~1.8916 GB
  assert.ok(Math.abs(kv.sizeGB - 1.8916) < 0.001, `KV was ${kv.sizeGB}, expected ~1.8916`);
});

test('GLM-5.3-Flash KDA recurrent state: 34 layers, F32, 1+rs_seq rows', () => {
  const arch = extractArch(GLM53F_CONFIG);
  // per layer: conv 3*(4-1)*64*128 = 73728 + S 64*128*128 = 1048576 = 1122304 elems
  const bytesPerLayer = (3 * 3 * 64 * 128 + 64 * 128 * 128) * 4; // f32
  const rs = calcRecurrentState(arch, 0);
  assert.equal(rs.nRecurLayers, 34);
  assert.equal(rs.nAttnLayers, 11);
  assert.ok(Math.abs(rs.sizeGB - 34 * bytesPerLayer / 1e9) < 1e-9, `rs ${rs.sizeGB}`);
  assert.ok(Math.abs(rs.sizeGB - 0.1524) < 0.001, `rs was ${rs.sizeGB}, expected ~0.1524`);
  // MTP on: n_rs_seq=2 -> 3x the rows
  const rsMtp = calcRecurrentState(arch, 2);
  assert.ok(Math.abs(rsMtp.sizeGB - 3 * rs.sizeGB) < 1e-9);
});

test('GLM-5.3-Flash MTP draft KV: 1 NextN latent + its indexer (f16)', () => {
  const arch = extractArch(GLM53F_CONFIG);
  const ctx = 131072, mtpKbpp = 0.5625; // q4_0 draft
  const mtp = calcMtpKV(ctx, arch, mtpKbpp, 0);
  const kBytes = 1 * 512 * ctx * mtpKbpp;
  const idxBytes = 1 * 3 * 128 * ctx * 2.0; // NextN layer also reads the indexer
  assert.equal(mtp.vGB, 0);
  assert.ok(Math.abs(mtp.sizeGB - (kBytes + idxBytes) / 1e9) < 1e-9, `MTP was ${mtp.sizeGB}`);
  // latent 0.0377 + indexer 0.1007 = ~0.1384 GB
  assert.ok(Math.abs(mtp.sizeGB - 0.1384) < 0.001, `MTP was ${mtp.sizeGB}, expected ~0.1384`);
});

test('withMtpDisabled drops GLM-5.3-Flash NextN layer and draft KV', () => {
  const arch = extractArch(GLM53F_CONFIG);
  const noMtp = withMtpDisabled(arch);
  assert.equal(noMtp.numLayers, 45);
  const ctx = 131072, bpp = 1.0625;
  assert.ok(Math.abs(calcKVCache(ctx, 512, arch, bpp, bpp).sizeGB
                  - calcKVCache(ctx, 512, noMtp, bpp, bpp).sizeGB) < 1e-9);
  assert.equal(calcMtpKV(ctx, noMtp, bpp, 0).sizeGB, 0);
  // Disabling MTP also drops n_rs_seq 2 -> 0: same 34 KDA layers, 3 rows -> 1 row
  const rsMtpOn = calcRecurrentState(arch, 2);
  const rsMtpOff = calcRecurrentState(noMtp, 0);
  assert.equal(rsMtpOff.nRecurLayers, rsMtpOn.nRecurLayers);
  assert.ok(Math.abs(rsMtpOff.sizeGB - rsMtpOn.sizeGB / 3) < 1e-9);
});

// ---------------------------------------------------------------------------
// estimateVram end-to-end
// ---------------------------------------------------------------------------

test('estimateVram aggregates per-quant totals with KV + compute', () => {
  const ggufFiles = [
    { path: 'model-UD-Q4_K_XL.gguf', size: 16e9 },
    { path: 'model-UD-Q8_K_XL.gguf', size: 32e9 },
  ];
  const est = estimateVram({
    ggufFiles,
    ctx: 131072, ubatch: 512,
    arch: GLIMMER_ARCH,
    kBpp: 1.0625, vBpp: 1.0625,
    mtpkBpp: 0.5625, mtpvBpp: 0.5625,
  });

  assert.equal(est.quants.length, 2);
  assert.equal(est.quants[0].quant, 'UD-Q8_K_XL'); // sorted highest first
  assert.equal(est.quants[1].quant, 'UD-Q4_K_XL');
  // Each quant total should include weights + KV + compute
  const q4 = est.quants[1];
  assert.ok(q4.total > q4.sizeGB, 'total should include more than just weights');
  assert.ok(Math.abs(q4.total - (16 + 0.98 + est.computeBuf)) < 0.1, `Q4 total was ${q4.total}`);
});

test('estimateVram includes DFlash sidecar in totals', () => {
  const ggufFiles = [
    { path: 'model-UD-Q4_K_XL.gguf', size: 16e9 },
  ];
  const dflashArch = {
    numLayers: 5, mtpLayers: 0,
    numKVHeads: 8, headDim: 128,
    slidingWindow: 2048,
    layerTypes: ['sliding_attention','sliding_attention','sliding_attention','sliding_attention','sliding_attention'],
    fullAttnInterval: 1, isHybrid: false, vocabSize: 202048,
  };
  const est = estimateVram({
    ggufFiles,
    ctx: 131072, ubatch: 512,
    arch: GLIMMER_ARCH,
    kBpp: 1.0625, vBpp: 1.0625,
    mtpkBpp: 0.5625, mtpvBpp: 0.5625,
    dflash: { sizeGB: 1.52, arch: dflashArch },
  });

  assert.ok(est.dflash.total > 1.5, `DFlash total was ${est.dflash.total}, expected > 1.5 (weights + KV)`);
  assert.ok(est.dflash.kv && est.dflash.kv.sizeGB > 0, 'DFlash KV should be computed');
  // Total should include dflash
  const q4 = est.quants[0];
  assert.ok(q4.total > 16 + 1.5 + 0.9, 'total should include DFlash');
});

test('estimateVram uses independent dflash K/V BPP when provided', () => {
  const ggufFiles = [{ path: 'model-UD-Q4_K_XL.gguf', size: 16e9 }];
  const dflashArch = {
    numLayers: 5, mtpLayers: 0,
    numKVHeads: 8, headDim: 128,
    slidingWindow: 2048,
    layerTypes: ['sliding_attention','sliding_attention','sliding_attention','sliding_attention','sliding_attention'],
    fullAttnInterval: 1, isHybrid: false, vocabSize: 202048,
  };
  const estMain = estimateVram({
    ggufFiles, ctx: 131072, ubatch: 512, arch: GLIMMER_ARCH,
    kBpp: 1.0625, vBpp: 1.0625, mtpkBpp: 0.5625, mtpvBpp: 0.5625,
    dflash: { sizeGB: 1.52, arch: dflashArch },
  });
  const estDraft = estimateVram({
    ggufFiles, ctx: 131072, ubatch: 512, arch: GLIMMER_ARCH,
    kBpp: 1.0625, vBpp: 1.0625, mtpkBpp: 0.5625, mtpvBpp: 0.5625,
    dflash: { sizeGB: 1.52, arch: dflashArch },
    dflashkBpp: 0.5625, dflashvBpp: 0.5625, // q4_0 instead of q8_0
  });
  // Draft KV with lower bpp should be smaller than main-bpp KV
  assert.ok(estDraft.dflash.kv.sizeGB < estMain.dflash.kv.sizeGB,
    `DFlash draft KV at q4_0 (${estDraft.dflash.kv.sizeGB}) should be < q8_0 (${estMain.dflash.kv.sizeGB})`);
  assert.ok(estDraft.dflash.total < estMain.dflash.total,
    `DFlash total at q4_0 (${estDraft.dflash.total}) should be < q8_0 (${estMain.dflash.total})`);
});

// ---------------------------------------------------------------------------
// Gemma4 MTP sidecar compute buffer (shared KV, zero draft KV)
// ---------------------------------------------------------------------------

const GEMMA4_MTP_SIDECAR_ARCH = {
  numLayers: 4,
  hiddenSize: 1024,
  nEmbdOut: 2816,
  sharedKvLayers: 4, // all layers share KV — zero draft KV
  vocabSize: 262144,
};

test('calcMtpDraftCompute returns 0 for null arch', () => {
  assert.equal(calcMtpDraftCompute(512, null), 0);
});

test('calcMtpDraftCompute estimates logits + base for gemma4-assistant', () => {
  const compute = calcMtpDraftCompute(512, GEMMA4_MTP_SIDECAR_ARCH);
  // logits = 262144 * 512 * 2 = 0.268 GB, base = 0.25 GB -> ~0.52 GB
  assert.ok(compute > 0.4, `compute was ${compute}, expected > 0.4`);
  assert.ok(compute < 0.7, `compute was ${compute}, expected < 0.7`);
});

test('estimateVram includes MTP sidecar weights + compute (no KV)', () => {
  const ggufFiles = [{ path: 'model-UD-Q4_K_XL.gguf', size: 16e9 }];
  const est = estimateVram({
    ggufFiles,
    ctx: 131072, ubatch: 512,
    arch: GLIMMER_ARCH,
    kBpp: 1.0625, vBpp: 1.0625,
    mtpkBpp: 0.5625, mtpvBpp: 0.5625,
    mtpSidecar: { sizeGB: 0.46, arch: GEMMA4_MTP_SIDECAR_ARCH },
  });

  assert.ok(est.mtpSidecar, 'mtpSidecar should be present in result');
  assert.ok(est.mtpSidecar.total > 0.46, `total was ${est.mtpSidecar.total}, expected > 0.46 (weights + compute)`);
  assert.ok(est.mtpSidecar.computeGB > 0, 'compute buffer should be non-zero');
  // Total = weights (0.46) + compute (~0.52), so ~0.98
  assert.ok(Math.abs(est.mtpSidecar.total - (0.46 + est.mtpSidecar.computeGB)) < 0.01, 'total should be weights + compute');

  // Per-quant total should include the sidecar cost
  const q4 = est.quants[0];
  assert.ok(q4.total > 16 + 0.9, 'per-quant total should include MTP sidecar');
});

test('estimateVram with no mtpSidecar returns null', () => {
  const ggufFiles = [{ path: 'model-Q4_K_M.gguf', size: 16e9 }];
  const est = estimateVram({
    ggufFiles, ctx: 8192, ubatch: 512,
    arch: GLIMMER_ARCH,
    kBpp: 1.0625, vBpp: 1.0625,
  });
  assert.equal(est.mtpSidecar, null);
});

// ---------------------------------------------------------------------------
// mmproj compute buffer (CLIP vision encoder)
// ---------------------------------------------------------------------------

test('calcMmprojCompute estimates CLIP vision encoder compute buffer', () => {
  // Real params from Muse-Glimmer mmproj (extractClipArch)
  const clipArch = {
    nEmbmd: 1536, nLayer: 50, nFf: 8960,
    imageSize: 896, patchSize: 14, nPatches: 4096,
  };
  const compute = calcMmprojCompute(clipArch);
  // Should be non-trivial but reasonable (~0.2 GB peak activations)
  assert.ok(compute > 0.1, `compute was ${compute}, expected > 0.1`);
  assert.ok(compute < 0.5, `compute was ${compute}, expected < 0.5`);
});

test('calcMmprojCompute returns 0 for null or incomplete clipArch', () => {
  assert.equal(calcMmprojCompute(null), 0);
  assert.equal(calcMmprojCompute({}), 0);
  assert.equal(calcMmprojCompute({ nPatches: 100, nLayer: 0, nEmbmd: 1024, nFf: 4096 }), 0);
  assert.equal(calcMmprojCompute({ nPatches: 0, nLayer: 10, nEmbmd: 1024, nFf: 4096 }), 0);
});
