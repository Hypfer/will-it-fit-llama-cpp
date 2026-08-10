/**
 * @typedef {Object} CommandConfig
 * @property {string} repo              - HF repo ID ("org/model")
 * @property {number} ctx               - context length
 * @property {number} ubatch            - physical batch size (n_ubatch)
 * @property {string} kCacheType        - e.g. "q8_0", "fp16"
 * @property {string} vCacheType        - e.g. "q8_0", "fp16"
 * @property {object|null} arch         - main model architecture (from extractArch)
 * @property {object|null} mtp          - MTP options {kCacheType, vCacheType} or null
 * @property {object|null} dflash       - DFlash sidecar {fileName, kCacheType, vCacheType} or null
 * @property {object|null} mmproj       - mmproj options {fileName, cpuOnly} or null
 */

/**
 * Build the llama-server command string.
 *
 * @param {CommandConfig} cfg
 * @returns {string} the full command, ready to copy-paste
 */
export function buildCommand(cfg) {
  const { repo, ctx, ubatch, kCacheType, vCacheType, arch, mtp, dflash, mmproj } = cfg;

  const modelPlaceholder = repo.split('/').pop().replace(/-GGUF$/i, '');

  let cmd = `llama-server -m ./models/<${modelPlaceholder}_Q...>.gguf -c ${ctx} -ub ${ubatch} -ngl 999 -fa --jinja`;

  if (kCacheType !== 'fp16') cmd += ` --cache-type-k ${kCacheType}`;
  if (vCacheType !== 'fp16') cmd += ` --cache-type-v ${vCacheType}`;

  // MTP speculative decoding (embedded head in the model)
  if (arch && arch.mtpLayers > 0) {
    cmd += ` --spec-type draft-mtp --spec-draft-n-max 2`;
    if (mtp && mtp.kCacheType !== 'fp16') cmd += ` --cache-type-k-draft ${mtp.kCacheType}`;
    if (mtp && mtp.vCacheType !== 'fp16') cmd += ` --cache-type-v-draft ${mtp.vCacheType}`;
  }

  // DFlash speculative decoding (separate sidecar draft model)
  if (dflash && dflash.fileName) {
    cmd += ` --spec-type draft-dflash --spec-draft-n-max 2`;
    cmd += ` -md ./models/<${dflash.fileName}>`;
    if (dflash.kCacheType && dflash.kCacheType !== 'fp16') cmd += ` --cache-type-k-draft ${dflash.kCacheType}`;
    if (dflash.vCacheType && dflash.vCacheType !== 'fp16') cmd += ` --cache-type-v-draft ${dflash.vCacheType}`;
  }

  // Vision projector (mmproj)
  if (mmproj && mmproj.fileName) {
    cmd += ` --mmproj ./models/<${mmproj.fileName}>`;
    if (mmproj.cpuOnly) cmd += ` --no-mmproj-offload`;
  }

  // n-gram mod speculative decoding (draftless, always enabled)
  // Must come after --spec-type since it appends to the types list.
  // (arg.cpp: --spec-default does push_back, --spec-type does assignment)
  cmd += ` --spec-default`;

  return cmd;
}
