import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseGgufHeader, extractDflashArch, extractClipArch, extractMtpArch } from '../src/js/gguf-parser.js';

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
