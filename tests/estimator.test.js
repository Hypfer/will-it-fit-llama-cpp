import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  QUANT_ORDER, sortQuants, parseQuant, isFirstPart, normalizeRepo,
  effectiveBpp, extractArch, calcKVCache, calcMtpKV, calcComputeBuffer,
  calcMmprojCompute, calcMtpDraftCompute, estimateVram,
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
