import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseGgufHeader, extractDflashArch, extractClipArch, extractMtpArch, ggufToConfig } from '../src/js/gguf-parser.js';

// Use the real GGUF header bytes (committed as test fixtures)
const __dirname = dirname(fileURLToPath(import.meta.url));
const HEADER_BYTES = readFileSync(join(__dirname, 'fixtures', 'dflash_header.bin'));
const MMPROJ_BYTES = readFileSync(join(__dirname, 'fixtures', 'mmproj_header.bin'));
const MTP_SIDECAR_BYTES = readFileSync(join(__dirname, 'fixtures', 'mtp_sidecar_header.bin'));
const MTP_SIDECAR_NEXTN_BYTES = readFileSync(join(__dirname, 'fixtures', 'mtp_sidecar_nextn_header.bin'));

test('parseGgufHeader reads magic and version', () => {
  const { magic, version, nTensors, nKV } = parseGgufHeader(HEADER_BYTES);
  assert.equal(magic, 0x46554747); // "GGUF"
  assert.equal(version, 3);
  assert.equal(nTensors, 58);
  assert.ok(nKV >= 20, `expected at least 20 KV pairs, got ${nKV}`);
});

test('parseGgufHeader reads dflash architecture metadata', () => {
  const { metadata } = parseGgufHeader(HEADER_BYTES);
  assert.equal(metadata['general.architecture'], 'dflash');
  assert.equal(metadata['general.name'], 'Hf_Museglimmer');
  assert.equal(metadata['general.size_label'], '2.6B');
});

test('parseGgufHeader reads dflash attention params', () => {
  const { metadata } = parseGgufHeader(HEADER_BYTES);
  assert.equal(metadata['dflash.block_count'], 5);
  assert.equal(metadata['dflash.attention.head_count'], 32);
  assert.equal(metadata['dflash.attention.head_count_kv'], 8);
  assert.equal(metadata['dflash.attention.key_length'], 128);
  assert.equal(metadata['dflash.attention.value_length'], 128);
  assert.equal(metadata['dflash.context_length'], 131072);
  assert.equal(metadata['dflash.attention.sliding_window'], 2048);
  assert.equal(metadata['dflash.block_size'], 16);
});

test('parseGgufHeader reads target_layers array', () => {
  const { metadata } = parseGgufHeader(HEADER_BYTES);
  const targetLayers = metadata['dflash.target_layers'];
  assert.ok(Array.isArray(targetLayers), 'target_layers should be an array');
  assert.deepEqual(targetLayers, [2, 14, 26, 38, 50]);
});

test('parseGgufHeader skips large tokenizer token arrays', () => {
  const { metadata } = parseGgufHeader(HEADER_BYTES);
  // tokenizer.ggml.tokens would be a huge array, should be skipped
  // but tokenizer.ggml.model (a string) should be present
  assert.equal(metadata['tokenizer.ggml.model'], 'gpt2');
  assert.equal(metadata['tokenizer.ggml.pre'], 'llama4');
});

test('extractDflashArch builds architecture object', () => {
  const { metadata } = parseGgufHeader(HEADER_BYTES);
  const arch = extractDflashArch(metadata);
  assert.equal(arch.numLayers, 5);
  assert.equal(arch.numKVHeads, 8);
  assert.equal(arch.headDim, 128);
  assert.equal(arch.slidingWindow, 2048);
  // n_embd_k_gqa = 8 * 128 = 1024
  assert.equal(arch.numKVHeads * arch.headDim, 1024);
  // All 5 layers are SWA
  assert.ok(arch.layerTypes);
  assert.equal(arch.layerTypes.length, 5);
  assert.deepEqual(arch.layerTypes, [
    'sliding_attention', 'sliding_attention', 'sliding_attention',
    'sliding_attention', 'sliding_attention',
  ]);
});

test('extractDflashArch returns null for non-dflash model', () => {
  assert.equal(extractDflashArch({ 'general.architecture': 'llama' }), null);
  assert.equal(extractDflashArch({}), null);
});

test('parseGgufHeader rejects non-GGUF buffer', () => {
  const fakeBuf = new ArrayBuffer(64);
  const dv = new DataView(fakeBuf);
  dv.setUint32(0, 0x12345678, true);
  assert.throws(() => parseGgufHeader(fakeBuf), /Not a GGUF file/);
});

// ---------------------------------------------------------------------------
// mmproj / CLIP header parsing
// ---------------------------------------------------------------------------

test('parseGgufHeader reads mmproj CLIP vision metadata', () => {
  const { metadata } = parseGgufHeader(MMPROJ_BYTES);
  assert.equal(metadata['clip.has_vision_encoder'], true);
  assert.equal(metadata['clip.projector_type'], 'muse-glimmer');
  assert.equal(metadata['clip.vision.image_size'], 896);
  assert.equal(metadata['clip.vision.patch_size'], 14);
  assert.equal(metadata['clip.vision.embedding_length'], 1536);
  assert.equal(metadata['clip.vision.feed_forward_length'], 8960);
  assert.equal(metadata['clip.vision.block_count'], 50);
});

test('extractClipArch builds vision encoder params', () => {
  const { metadata } = parseGgufHeader(MMPROJ_BYTES);
  const arch = extractClipArch(metadata);
  assert.equal(arch.nEmbmd, 1536);
  assert.equal(arch.nLayer, 50);
  assert.equal(arch.nFf, 8960);
  assert.equal(arch.imageSize, 896);
  assert.equal(arch.patchSize, 14);
  // nPatches = (896/14)^2 = 64^2 = 4096
  assert.equal(arch.nPatches, 4096);
});

test('extractClipArch returns null for non-CLIP metadata', () => {
  assert.equal(extractClipArch({ 'general.architecture': 'llama' }), null);
  assert.equal(extractClipArch({}), null);
});

// ---------------------------------------------------------------------------
// MTP draft sidecar header parsing (gemma4-assistant + generic nextn)
// ---------------------------------------------------------------------------

test('parseGgufHeader reads gemma4-assistant architecture metadata', () => {
  const { metadata } = parseGgufHeader(MTP_SIDECAR_BYTES);
  assert.equal(metadata['general.architecture'], 'gemma4-assistant');
});

test('parseGgufHeader reads gemma4-assistant MTP params', () => {
  const { metadata } = parseGgufHeader(MTP_SIDECAR_BYTES);
  assert.equal(metadata['gemma4-assistant.block_count'], 4);
  assert.equal(metadata['gemma4-assistant.embedding_length'], 1024);
  assert.equal(metadata['gemma4-assistant.embedding_length_out'], 2816);
  assert.equal(metadata['gemma4-assistant.attention.shared_kv_layers'], 4);
  assert.equal(metadata['gemma4-assistant.nextn_predict_layers'], 4);
});

test('parseGgufHeader reads gemma4-assistant attention params', () => {
  const { metadata } = parseGgufHeader(MTP_SIDECAR_BYTES);
  assert.equal(metadata['gemma4-assistant.attention.head_count'], 16);
  assert.deepEqual(metadata['gemma4-assistant.attention.head_count_kv'], [8, 8, 8, 2]);
  assert.equal(metadata['gemma4-assistant.attention.key_length'], 512);
  assert.equal(metadata['gemma4-assistant.attention.sliding_window'], 1024);
  assert.deepEqual(metadata['gemma4-assistant.attention.sliding_window_pattern'], [true, true, true, false]);
});

test('extractMtpArch builds architecture object from gemma4-assistant header', () => {
  const { metadata } = parseGgufHeader(MTP_SIDECAR_BYTES);
  const arch = extractMtpArch(metadata);
  assert.equal(arch.numLayers, 4);
  assert.equal(arch.hiddenSize, 1024);
  assert.equal(arch.nEmbdOut, 2816);
  assert.equal(arch.sharedKvLayers, 4);
  // All layers share KV — sharedKvLayers == numLayers means zero draft KV
  assert.equal(arch.sharedKvLayers, arch.numLayers);
});

test('extractMtpArch returns null for non-gemma4-assistant model', () => {
  assert.equal(extractMtpArch({ 'general.architecture': 'llama' }), null);
  assert.equal(extractMtpArch({ 'general.architecture': 'gemma4' }), null);
  assert.equal(extractMtpArch({}), null);
});

// Real header from unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf:
// a regular qwen35 arch whose nextn_predict_layers marks the MTP draft.
test('parseGgufHeader reads generic nextn sidecar architecture metadata', () => {
  const { metadata } = parseGgufHeader(MTP_SIDECAR_NEXTN_BYTES);
  assert.equal(metadata['general.architecture'], 'qwen35');
  assert.equal(metadata['qwen35.nextn_predict_layers'], 1);
});

test('extractMtpArch builds architecture object from a generic nextn sidecar header', () => {
  const { metadata } = parseGgufHeader(MTP_SIDECAR_NEXTN_BYTES);
  const arch = extractMtpArch(metadata);
  assert.equal(arch.numLayers, 1);
});

test('extractMtpArch returns null for arch without nextn metadata', () => {
  assert.equal(extractMtpArch({ 'general.architecture': 'qwen35', 'qwen35.block_count': 48 }), null);
});

// ---------------------------------------------------------------------------
// Truncated header handling (partial range fetch)
// ---------------------------------------------------------------------------
// Real Kolibri-1 header (Hob-forge/Kolibri-1-GGUF, Q4_K_M): 46 KVs, of which
// the 128k-token tokenizer arrays overrun the fetched range. The parser must
// return every KV before the tokenizer, never throw, and skip tensor info.

const KOLIBRI_BYTES = readFileSync(join(__dirname, 'fixtures', 'kolibri1_header.bin'));

test('parseGgufHeader survives a truncated tokenizer array (Kolibri-1)', () => {
  const { metadata, truncated, tensorInfos, nKV } = parseGgufHeader(KOLIBRI_BYTES);
  assert.equal(truncated, true);
  assert.equal(nKV, 46);
  assert.equal(tensorInfos.length, 0); // tensor info unreachable past the cut
  // Architecture keys come before the tokenizer — all must be present
  assert.equal(metadata['general.architecture'], 'kolibri1');
  assert.equal(metadata['kolibri1.block_count'], 50);
  assert.equal(metadata['kolibri1.attention.head_count_kv'], 4);
  assert.equal(metadata['kolibri1.attention.sliding_window'], 513);
  assert.ok(Math.abs(metadata['kolibri1.attention.layer_norm_rms_epsilon'] - 1e-6) < 1e-12);
  const pat = metadata['kolibri1.attention.sliding_window_pattern'];
  assert.ok(Array.isArray(pat) && pat.length === 50);
  assert.equal(pat.filter(Boolean).length, 40); // 40 SWA + 10 full
  assert.equal(metadata['kolibri1.expert_count'], 384);
  // The tokenizer array that killed the parser is marked, not thrown
  assert.ok(String(metadata['tokenizer.ggml.tokens']).includes('truncated'));
});

test('parseGgufHeader does not throw on a buffer cut mid-KV', () => {
  // Slice mid-way through the header: no exception, whatever fits is returned
  const short = KOLIBRI_BYTES.subarray(0, 800);
  const h = parseGgufHeader(short);
  assert.equal(h.magic, 0x46554747);
  assert.equal(h.truncated, true);
});

// ---------------------------------------------------------------------------
// ggufToConfig — GGUF-header → config fallback for extractArch()
// ---------------------------------------------------------------------------

test('ggufToConfig maps kolibri1 GGUF keys to config shape', () => {
  const { metadata } = parseGgufHeader(KOLIBRI_BYTES);
  const cfg = ggufToConfig(metadata);
  assert.equal(cfg.num_hidden_layers, 50);
  assert.equal(cfg.num_attention_heads, 48);
  assert.equal(cfg.num_key_value_heads, 4);
  assert.equal(cfg.head_dim, 128);
  assert.equal(cfg.sliding_window, 513);
  assert.equal(cfg.max_position_embeddings, 262144);
  // GGUF pattern: true = SWA layer → transformers sliding_attention
  assert.equal(cfg.layer_types.length, 50);
  assert.equal(cfg.layer_types[0], 'sliding_attention');
  assert.equal(cfg.layer_types[4], 'full_attention');
  assert.equal(cfg.layer_types.filter(t => t === 'sliding_attention').length, 40);
});

test('ggufToConfig returns null without architecture metadata', () => {
  assert.equal(ggufToConfig({}), null);
  assert.equal(ggufToConfig({ 'general.name': 'x' }), null);
});

// ---------------------------------------------------------------------------
// Inkling-2 Mini (ggml-org/Inkling-2-Mini-GGUF, IQ4_NL): real header of the
// inkling2_loop variant — per-layer head arrays, loop core, shortconv state
// ---------------------------------------------------------------------------

const INKLING_BYTES = readFileSync(join(__dirname, 'fixtures', 'inkling2_header.bin'));
// Real mmproj header of the same repo (BF16 vision tower)
const INKLING_MMPROJ_BYTES = readFileSync(join(__dirname, 'fixtures', 'inkling2_mmproj_header.bin'));

test('parseGgufHeader reads inkling2_loop architecture metadata', () => {
  const { metadata, nKV } = parseGgufHeader(INKLING_BYTES);
  assert.equal(metadata['general.architecture'], 'inkling2_loop');
  assert.equal(metadata['general.name'], 'Inkling 2 Mini');
  assert.equal(metadata['general.size_label'], '128x1.8B');
  assert.ok(nKV >= 45);
});

test('parseGgufHeader reads inkling2_loop hparams', () => {
  const { metadata } = parseGgufHeader(INKLING_BYTES);
  assert.equal(metadata['inkling2_loop.block_count'], 24);
  assert.equal(metadata['inkling2_loop.embedding_length'], 2304);
  assert.equal(metadata['inkling2_loop.context_length'], 1048576);
  assert.equal(metadata['inkling2_loop.vocab_size'], 201024);       // padded
  assert.equal(metadata['inkling2_loop.vocab_size_unpadded'], 200058);
  assert.equal(metadata['inkling2_loop.attention.key_length'], 128);
  assert.equal(metadata['inkling2_loop.attention.key_length_swa'], 128);
  assert.equal(metadata['inkling2_loop.attention.sliding_window'], 1024);
  assert.equal(metadata['inkling2_loop.shortconv.l_cache'], 4);
  assert.equal(metadata['inkling2_loop.attn_res.block_size'], 6);
  assert.equal(metadata['inkling2_loop.expert_count'], 128);
  assert.equal(metadata['inkling2_loop.expert_used_count'], 8);
});

test('parseGgufHeader reads per-layer head arrays and the layer pattern', () => {
  const { metadata } = parseGgufHeader(INKLING_BYTES);
  // The converter writes per-layer head counts (SWA variants resolved per layer)
  const heads = metadata['inkling2_loop.attention.head_count'];
  const kvHeads = metadata['inkling2_loop.attention.head_count_kv'];
  assert.ok(Array.isArray(heads) && heads.length === 24);
  assert.ok(heads.every(h => h === 24));
  assert.ok(Array.isArray(kvHeads) && kvHeads.length === 24);
  assert.ok(kvHeads.every(h => h === 8));
  // 20 SWA layers, full attention at 5, 11, 17, 23
  const pat = metadata['inkling2_loop.attention.sliding_window_pattern'];
  assert.equal(pat.filter(Boolean).length, 20);
  for (const i of [5, 11, 17, 23]) assert.equal(pat[i], false);
});

test('parseGgufHeader reads the loop core config', () => {
  const { metadata } = parseGgufHeader(INKLING_BYTES);
  assert.equal(metadata['inkling2_loop.loop.start'], 6);
  assert.equal(metadata['inkling2_loop.loop.end'], 18);
  assert.equal(metadata['inkling2_loop.loop.count'], 3);
  assert.equal(metadata['inkling2_loop.loop.gate_rank'], 64);
  assert.equal(metadata['inkling2_loop.loop.varying'], false);
});

test('ggufToConfig maps inkling2_loop GGUF keys to config shape', () => {
  const { metadata } = parseGgufHeader(INKLING_BYTES);
  const cfg = ggufToConfig(metadata);
  assert.equal(cfg.model_type, 'inkling2_loop');
  assert.equal(cfg.num_hidden_layers, 24);
  assert.equal(cfg.hidden_size, 2304);
  assert.deepEqual(cfg.loop_config, { core_start: 6, core_end: 18, num_passes: 3 });
  assert.equal(cfg.sconv_kernel_size, 4);
  assert.equal(cfg.sliding_window_size, 1024);
  assert.equal(cfg.head_dim, 128);
  assert.equal(cfg.swa_head_dim, 128);
  assert.equal(cfg.vocab_size, 201024); // padded
  assert.equal(cfg.max_position_embeddings, 1048576);
  // per-layer arrays pass through for extractInkling2's positional resolution
  assert.ok(Array.isArray(cfg.num_key_value_heads) && cfg.num_key_value_heads.length === 24);
  // GGUF pattern: true = SWA -> sliding_attention
  assert.equal(cfg.layer_types.length, 24);
  assert.equal(cfg.layer_types.filter(t => t === 'sliding_attention').length, 20);
  assert.equal(cfg.layer_types[5], 'full_attention');
});

test('extractClipArch reads the Inkling-2 Mini vision tower', () => {
  const { metadata } = parseGgufHeader(INKLING_MMPROJ_BYTES);
  assert.equal(metadata['clip.vision.projector_type'], 'inkling2v');
  const clip = extractClipArch(metadata);
  assert.equal(clip.nEmbmd, 1024);
  assert.equal(clip.nLayer, 23);
  assert.equal(clip.nFf, 4096);
  assert.equal(clip.imageSize, 640);
  assert.equal(clip.patchSize, 20);
  // nPatches = (640/20)^2 = 1024
  assert.equal(clip.nPatches, 1024);
});

// ---------------------------------------------------------------------------
// unsloth/Inkling-2-Mini-GGUF root UD quants are loop models in unsloth's
// layout: arch "inkling2" with block_count ALREADY the expanded cache-slot
// count (48), loop tensors present, only blk.0..23 weights, and the loop
// described by num_loops + loop.core_start/.core_end instead of loop.count.
// The original inkling2_loop layout (loop.count + physical block_count) lives
// in Mainline_Support/*.gguf_file — wrong extension, excluded by the app.
// The dspark- sidecar is a dflash-arch GGUF plus the Markov/confidence heads.
// ---------------------------------------------------------------------------

const INKLING_UNROLLED_BYTES = readFileSync(join(__dirname, 'fixtures', 'inkling2_unsloth_header.bin'));
const DSPARK_BYTES = readFileSync(join(__dirname, 'fixtures', 'dspark_header.bin'));

test('parseGgufHeader reads the unsloth loop layout (pre-expanded block_count)', () => {
  const { metadata } = parseGgufHeader(INKLING_UNROLLED_BYTES);
  assert.equal(metadata['general.architecture'], 'inkling2');
  // Loop model in unsloth's layout: slot count baked into block_count, loop
  // described by num_loops + core_start/core_end (no loop.count)
  assert.equal(metadata['inkling2.block_count'], 48);
  assert.equal(metadata['inkling2.num_loops'], 3);
  assert.equal(metadata['inkling2.loop.core_start'], 6);
  assert.equal(metadata['inkling2.loop.core_end'], 18);
  assert.equal(metadata['inkling2.loop.count'], undefined);
  // Pattern already covers all 48 slots
  const pat = metadata['inkling2.attention.sliding_window_pattern'];
  assert.equal(pat.length, 48);
  assert.equal(pat.filter(Boolean).length, 40);
});

test('ggufToConfig maps the unsloth layout as pre_expanded (never re-expands)', () => {
  const { metadata } = parseGgufHeader(INKLING_UNROLLED_BYTES);
  const cfg = ggufToConfig(metadata);
  assert.equal(cfg.model_type, 'inkling2');
  assert.equal(cfg.num_hidden_layers, 48);
  assert.deepEqual(cfg.loop_config, { core_start: 6, core_end: 18, num_passes: 3, pre_expanded: true });
  assert.equal(cfg.layer_types.length, 48);
  assert.equal(cfg.layer_types.filter(t => t === 'sliding_attention').length, 40);
  assert.equal(cfg.sconv_kernel_size, 4);
  assert.equal(cfg.vocab_size, 201024);
});

test('extractDflashArch reads the dspark sidecar (dflash arch + markov heads)', () => {
  const { metadata } = parseGgufHeader(DSPARK_BYTES);
  // The sidecar is still a dflash-arch GGUF — the variant comes from the
  // tensors (llama_model_dflash_variant), the metadata keys are all dflash.*
  assert.equal(metadata['general.architecture'], 'dflash');
  assert.equal(metadata['dflash.block_count'], 3);
  assert.equal(metadata['dflash.block_size'], 8);
  assert.equal(metadata['dflash.attention.head_count'], 24);
  assert.equal(metadata['dflash.attention.head_count_kv'], 8);
  assert.equal(metadata['dflash.attention.key_length'], 128);
  assert.equal(metadata['dflash.attention.sliding_window'], 1024);
  const arch = extractDflashArch(metadata);
  assert.equal(arch.numLayers, 3);
  assert.equal(arch.numKVHeads, 8);
  assert.equal(arch.headDim, 128);
  assert.equal(arch.slidingWindow, 1024);
  assert.deepEqual(arch.layerTypes, ['sliding_attention', 'full_attention', 'sliding_attention']);
});
