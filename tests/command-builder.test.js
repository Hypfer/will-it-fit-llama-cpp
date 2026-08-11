import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildCommand } from '../src/js/command-builder.js';

test('buildCommand generates basic command', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Foo-27B-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 }, mtp: null, dflash: null,
  });
  assert.ok(cmd.startsWith('llama-server -m ./models/<Foo-27B_Q...>.gguf'));
  assert.ok(cmd.includes('-c 8192'));
  assert.ok(cmd.includes('-ub 512'));
  assert.ok(cmd.includes('--cache-type-k q8_0'));
  assert.ok(cmd.includes('--cache-type-v q8_0'));
  assert.ok(cmd.includes('-ngl 999'));
  assert.ok(cmd.includes('-fa'));
  assert.ok(cmd.includes('--jinja'));
  // No --spec-type (no draft model), but --spec-default is always present
  assert.ok(!cmd.includes('--spec-type'));
  assert.ok(cmd.includes('--spec-default'));
});

test('buildCommand omits cache-type when fp16', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 4096, ubatch: 512,
    kCacheType: 'fp16', vCacheType: 'fp16',
    arch: { mtpLayers: 0 }, mtp: null, dflash: null,
  });
  assert.ok(!cmd.includes('--cache-type-k'));
  assert.ok(!cmd.includes('--cache-type-v'));
});

test('buildCommand adds MTP flags when arch has mtpLayers', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Foo-MTP-GGUF',
    ctx: 131072, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 1 },
    mtp: { kCacheType: 'q4_0', vCacheType: 'q4_0' },
    dflash: null,
  });
  assert.ok(cmd.includes('--spec-type draft-mtp'));
  assert.ok(cmd.includes('--spec-draft-n-max 2'));
  assert.ok(cmd.includes('--cache-type-k-draft q4_0'));
  assert.ok(cmd.includes('--cache-type-v-draft q4_0'));
});

test('buildCommand adds DFlash flags when sidecar detected', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Muse-Glimmer-30B-GGUF',
    ctx: 131072, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 },
    mtp: null,
    dflash: { fileName: 'dflash-kquant.gguf', kCacheType: 'q4_0', vCacheType: 'q4_0' },
  });
  assert.ok(cmd.includes('--spec-type draft-dflash'));
  assert.ok(cmd.includes('--spec-draft-n-max 2'));
  assert.ok(cmd.includes('-md ./models/<dflash-kquant.gguf>'));
  assert.ok(cmd.includes('--cache-type-k-draft q4_0'));
  assert.ok(cmd.includes('--cache-type-v-draft q4_0'));
  // Should NOT have MTP flags
  assert.ok(!cmd.includes('draft-mtp'));
});

test('buildCommand handles both MTP and DFlash (MTP takes precedence in flags)', () => {
  // In practice a model won't have both, but test the function doesn't crash
  const cmd = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 1 },
    mtp: { kCacheType: 'q4_0', vCacheType: 'q4_0' },
    dflash: { fileName: 'dflash-kquant.gguf' },
  });
  // Both spec types would be present (caller should avoid this)
  assert.ok(cmd.includes('draft-mtp'));
  assert.ok(cmd.includes('draft-dflash'));
});

test('buildCommand adds MTP sidecar flags (draft-mtp + -md, no cache-type-draft)', () => {
  const cmd = buildCommand({
    repo: 'unsloth/gemma-4-26B-A4B-it-GGUF',
    ctx: 131072, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 },
    mtp: null,
    dflash: null,
    mtpSidecar: { fileName: 'mtp-gemma-4-26B-A4B-it.gguf' },
  });
  assert.ok(cmd.includes('--spec-type draft-mtp'), 'should have draft-mtp spec type');
  assert.ok(cmd.includes('--spec-draft-n-max 2'), 'should have spec-draft-n-max 2');
  assert.ok(cmd.includes('-md ./models/<mtp-gemma-4-26B-A4B-it.gguf>'), 'should point -md at sidecar');
  // Shared KV: no draft cache-type flags
  assert.ok(!cmd.includes('--cache-type-k-draft'), 'should NOT have cache-type-k-draft (shared KV)');
  assert.ok(!cmd.includes('--cache-type-v-draft'), 'should NOT have cache-type-v-draft (shared KV)');
});

test('buildCommand MTP sidecar is mutually exclusive with embedded MTP', () => {
  // When arch.mtpLayers > 0 (embedded head), the sidecar branch is suppressed
  const cmd = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 1 },
    mtp: { kCacheType: 'q4_0', vCacheType: 'q4_0' },
    mtpSidecar: { fileName: 'mtp-sidecar.gguf' },
  });
  // Embedded MTP should fire, sidecar should NOT
  assert.ok(cmd.includes('draft-mtp'));
  assert.ok(cmd.includes('--cache-type-k-draft q4_0'), 'embedded MTP cache flags should be present');
  assert.ok(!cmd.includes('-md ./models/<mtp-sidecar.gguf>'), 'sidecar -md should be suppressed');
});

test('buildCommand adds mmproj flags for GPU offload', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 }, mtp: null, dflash: null,
    mmproj: { fileName: 'mmproj-kquant.gguf', cpuOnly: false },
  });
  assert.ok(cmd.includes('--mmproj ./models/<mmproj-kquant.gguf>'));
  assert.ok(!cmd.includes('--no-mmproj-offload'));
});

test('buildCommand adds --no-mmproj-offload for CPU mode', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 }, mtp: null, dflash: null,
    mmproj: { fileName: 'mmproj-kquant.gguf', cpuOnly: true },
  });
  assert.ok(cmd.includes('--mmproj ./models/<mmproj-kquant.gguf>'));
  assert.ok(cmd.includes('--no-mmproj-offload'));
});

test('buildCommand omits mmproj when null', () => {
  const cmd = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 }, mtp: null, dflash: null,
    mmproj: null,
  });
  assert.ok(!cmd.includes('--mmproj'));
  assert.ok(!cmd.includes('--no-mmproj-offload'));
});

test('buildCommand always includes --spec-default after --spec-type', () => {
  // --spec-default must come after --spec-type because it appends to the
  // speculative types list, while --spec-type replaces it (arg.cpp:4556 push_back
  // vs 546 assignment). If the order is reversed, --spec-type wipes ngram-mod.
  const cmdWithMtp = buildCommand({
    repo: 'unsloth/Foo-MTP-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 1 },
    mtp: { kCacheType: 'q4_0', vCacheType: 'q4_0' },
    dflash: null,
  });
  assert.ok(cmdWithMtp.includes('--spec-default'));
  assert.ok(cmdWithMtp.indexOf('--spec-type') < cmdWithMtp.indexOf('--spec-default'));

  // Even with no draft model at all, --spec-default is present
  const cmdNoDraft = buildCommand({
    repo: 'unsloth/Foo-GGUF',
    ctx: 8192, ubatch: 512,
    kCacheType: 'q8_0', vCacheType: 'q8_0',
    arch: { mtpLayers: 0 }, mtp: null, dflash: null,
  });
  assert.ok(cmdNoDraft.includes('--spec-default'));
});
