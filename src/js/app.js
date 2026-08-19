import {
  QUANT_ORDER, sortQuants, parseQuant, isFirstPart, normalizeRepo,
  extractArch, estimateVram, calcMmprojCompute, withMtpDisabled,
} from './vram-estimator.js';
import { parseGgufHeader, extractDflashArch, extractClipArch, extractMtpArch } from './gguf-parser.js';
import { buildCommand } from './command-builder.js';

// ---------------------------------------------------------------------------
// Module-level cache — only re-fetch on repo change
// ---------------------------------------------------------------------------
let cachedGGUF = null;
let cachedFiles = null;
let cachedArch = null;
let cachedDflash = null;   // { fileName, sizeGB, arch } or null
let cachedMmproj = null;   // { fileName, sizeGB, clipArch } or null (best variant)
let cachedMtpSidecar = null; // { fileName, sizeGB, arch } or null

const HF_BASE = 'https://huggingface.co';

// ---------------------------------------------------------------------------
// HF API helpers
// ---------------------------------------------------------------------------
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HF API ${r.status}: ${url}`);
  return r.json();
}

async function getRepoFiles(repoId) {
  return fetchJSON(`${HF_BASE}/api/models/${repoId}/tree/main?recursive=true`);
}

async function getGGUFInfo(repoId) {
  const meta = await fetchJSON(`${HF_BASE}/api/models/${repoId}?expand[]=gguf`);
  return meta.gguf || null;
}

async function getBaseModelConfig(repoId) {
  const info = await fetchJSON(`${HF_BASE}/api/models/${repoId}`);
  const baseTag = (info.tags || []).find(t => typeof t === 'string' && t.startsWith('base_model:'));
  if (!baseTag) return null;
  const baseModel = baseTag.replace('base_model:', '').split(':')[0];
  return fetchJSON(`${HF_BASE}/${baseModel}/raw/main/config.json`);
}

/**
 * Range-fetch the first N bytes of a GGUF file from HF and parse its header.
 * HF redirects to a CDN that honors Range requests + CORS.
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchGgufHeaderBytes(repoId, filePath, bytes = 131072) {
  const url = `${HF_BASE}/${repoId}/resolve/main/${filePath}`;
  const r = await fetch(url, { headers: { Range: `bytes=0-${bytes - 1}` } });
  if (!r.ok && r.status !== 206) throw new Error(`HF range fetch ${r.status}: ${filePath}`);
  return r.arrayBuffer();
}

// ---------------------------------------------------------------------------
// DFlash sidecar detection
// ---------------------------------------------------------------------------

/**
 * Find the DFlash sidecar in the repo file tree.
 * Detection rule (from unsloth PR #8338): root-level `dflash-*.gguf` files.
 */
function findDflashSidecar(files) {
  return files.find(f =>
    f.path.startsWith('dflash-') && f.path.endsWith('.gguf')
  );
}

/**
 * Detect + parse the DFlash sidecar: fetch its GGUF header and extract arch.
 * Sets `cachedDflash`. Called during loadModel().
 */
async function loadDflashSidecar(repoId, files) {
  const sidecar = findDflashSidecar(files);
  if (!sidecar) {
    cachedDflash = null;
    return null;
  }

  const fileName = sidecar.path.split('/').pop();
  const sizeGB = sidecar.size / 1e9;

  try {
    const buf = await fetchGgufHeaderBytes(repoId, sidecar.path);
    const { metadata } = parseGgufHeader(buf);
    const arch = extractDflashArch(metadata);
    cachedDflash = {
      fileName, sizeGB, arch, path: sidecar.path,
      headerError: arch ? null : 'No dflash.* metadata found in header',
    };
  } catch (e) {
    console.warn('DFlash header parse failed:', e);
    cachedDflash = {
      fileName, sizeGB, arch: null, path: sidecar.path,
      headerError: e.message,
    };
  }
  return cachedDflash;
}

// ---------------------------------------------------------------------------
// MTP sidecar detection
// ---------------------------------------------------------------------------

/**
 * Find the MTP sidecar in the repo file tree.
 * Detection: root-level `mtp-*.gguf` (unsloth convention, the prepacked Q8_0
 * default) is preferred; otherwise fall back to `MTP/mtp-*.gguf` (the larger
 * per-quant variants). Matches the DFlash root-level convention.
 */
function findMtpSidecar(files) {
  const rootLevel = files.find(f =>
    f.path.startsWith('mtp-') && f.path.endsWith('.gguf')
  );
  if (rootLevel) return rootLevel;
  return files.find(f =>
    f.path.startsWith('MTP/mtp-') && f.path.endsWith('.gguf')
  );
}

/**
 * Detect + parse the MTP sidecar: fetch its GGUF header and extract arch.
 * Sets `cachedMtpSidecar`. Called during loadModel().
 */
async function loadMtpSidecar(repoId, files) {
  const sidecar = findMtpSidecar(files);
  if (!sidecar) {
    cachedMtpSidecar = null;
    return null;
  }

  const fileName = sidecar.path.split('/').pop();
  const sizeGB = sidecar.size / 1e9;

  try {
    const buf = await fetchGgufHeaderBytes(repoId, sidecar.path);
    const { metadata } = parseGgufHeader(buf);
    const arch = extractMtpArch(metadata);
    cachedMtpSidecar = {
      fileName, sizeGB, arch, path: sidecar.path,
      headerError: arch ? null : 'No MTP draft metadata found in header',
    };
  } catch (e) {
    console.warn('MTP sidecar header parse failed:', e);
    cachedMtpSidecar = {
      fileName, sizeGB, arch: null, path: sidecar.path,
      headerError: e.message,
    };
  }
  return cachedMtpSidecar;
}

// ---------------------------------------------------------------------------
// mmproj sidecar detection
// ---------------------------------------------------------------------------

/**
 * Find all mmproj files in the repo file tree.
 * mmproj files start with "mmproj" (matching llama.cpp's find_best_sibling).
 */
function findMmprojFiles(files) {
  return files.filter(f => f.path.startsWith('mmproj') && f.path.endsWith('.gguf'));
}

// Preference order: kquant (smallest, calibrated) > Q8_0 > F16 > BF16 > F32 (anything else)
const MMPROJ_PREFERENCE = ['kquant', 'Q8_0', 'Q8', 'F16', 'BF16', 'F32'];

function pickBestMmproj(mmprojFiles) {
  if (!mmprojFiles.length) return null;
  for (const pref of MMPROJ_PREFERENCE) {
    const match = mmprojFiles.find(f => f.path.toUpperCase().includes(pref.toUpperCase()));
    if (match) return match;
  }
  // Fallback: smallest file
  return [...mmprojFiles].sort((a, b) => a.size - b.size)[0];
}

/**
 * Detect mmproj sidecar(s), pick the best variant, and parse its GGUF header
 * to extract CLIP vision encoder params for compute buffer estimation.
 * Sets `cachedMmproj`. Called during loadModel().
 */
async function loadMmprojSidecar(repoId, files) {
  const mmprojFiles = findMmprojFiles(files);
  const best = pickBestMmproj(mmprojFiles);
  if (!best) {
    cachedMmproj = null;
    return null;
  }
  cachedMmproj = {
    fileName: best.path.split('/').pop(),
    sizeGB: best.size / 1e9,
    path: best.path,
    clipArch: null,
    headerError: null,
    allVariants: mmprojFiles.map(f => ({ name: f.path.split('/').pop(), sizeGB: f.size / 1e9, path: f.path })),
  };

  // Range-fetch the header and parse CLIP vision encoder params
  try {
    const buf = await fetchGgufHeaderBytes(repoId, best.path);
    const { metadata } = parseGgufHeader(buf);
    cachedMmproj.clipArch = extractClipArch(metadata);
    if (!cachedMmproj.clipArch) {
      cachedMmproj.headerError = 'No clip.vision metadata found in header';
    }
  } catch (e) {
    console.warn('mmproj header parse failed:', e);
    cachedMmproj.headerError = e.message;
  }

  return cachedMmproj;
}
// ---------------------------------------------------------------------------
function fmtCtx(n) { return n.toLocaleString(); }

function quantLabel(selectEl) {
  return selectEl.selectedOptions[0].textContent.split(' ')[0];
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
async function loadModel() {
  const errorEl = document.getElementById('error');
  const metaEl = document.getElementById('meta');
  const resultsEl = document.getElementById('results');
  const cmdContainer = document.getElementById('cmd-container');
  errorEl.style.display = 'none';
  metaEl.innerHTML = '';
  resultsEl.innerHTML = '';
  cmdContainer.innerHTML = '';

  const repoInput = document.getElementById('repo');
  const repo = normalizeRepo(repoInput.value);
  repoInput.value = repo;
  if (!repo) {
    errorEl.textContent = 'Enter a repo ID';
    errorEl.style.display = 'block';
    return;
  }

  const btn = document.getElementById('load');
  btn.textContent = 'Loading metadata...';
  btn.disabled = true;

  try {
    const wasCached = !!cachedGGUF;
    cachedGGUF = await getGGUFInfo(repo);
    cachedFiles = await getRepoFiles(repo);

    let arch = null;
    try {
      const baseConfig = await getBaseModelConfig(repo);
      if (baseConfig) arch = extractArch(baseConfig);
    } catch (e) { console.warn('Config fetch failed:', e); }
    cachedArch = arch;

    // Detect + parse DFlash sidecar
    await loadDflashSidecar(repo, cachedFiles);

      // Detect mmproj sidecar
      await loadMmprojSidecar(repo, cachedFiles);

    // Detect + parse MTP sidecar
    await loadMtpSidecar(repo, cachedFiles);

    // MTP drafting section: one section, two shapes. Embedded MTP layers (from
    // the base config.json) when present; otherwise the sidecar draft file some
    // repos ship for small quants. The KV quant dropdowns only apply to the
    // embedded path — the sidecar shares the target's KV cache.
    const mtpSection = document.getElementById('mtp-section');
    const hasEmbeddedMtp = !!(arch && arch.mtpLayers > 0);
    if (mtpSection) {
      mtpSection.style.display = (hasEmbeddedMtp || cachedMtpSidecar) ? 'block' : 'none';
      const info = document.getElementById('mtp-info');
      if (hasEmbeddedMtp) {
        if (info) {
          info.textContent = `${arch.mtpLayers} MTP layer${arch.mtpLayers > 1 ? 's' : ''} (per base model config)`;
        }
      } else if (cachedMtpSidecar && info) {
        const url = `${HF_BASE}/${repo}/blob/main/${cachedMtpSidecar.path}`;
        let txt = `<a href="${url}" target="_blank" class="sidecar-link">${cachedMtpSidecar.fileName}</a> (${cachedMtpSidecar.sizeGB.toFixed(2)} GB)`;
        if (cachedMtpSidecar.arch) {
          txt += ` — ${cachedMtpSidecar.arch.numLayers} MTP layer${cachedMtpSidecar.arch.numLayers > 1 ? 's' : ''}`;
        } else if (cachedMtpSidecar.headerError) {
          txt += ` — header parse failed: ${cachedMtpSidecar.headerError}`;
        }
        info.innerHTML = txt;
      }
      const kvQuants = document.getElementById('mtp-kv-quants');
      if (kvQuants) kvQuants.style.display = hasEmbeddedMtp ? 'block' : 'none';
    }

    // Show/hide DFlash section
    const dflashSection = document.getElementById('dflash-section');
    if (dflashSection) {
      dflashSection.style.display = cachedDflash ? 'block' : 'none';
      if (cachedDflash) {
        const info = document.getElementById('dflash-info');
        if (info) {
          const url = `${HF_BASE}/${repo}/blob/main/${cachedDflash.path}`;
          let txt = `<a href="${url}" target="_blank" class="sidecar-link">${cachedDflash.fileName}</a> (${cachedDflash.sizeGB.toFixed(1)} GB)`;
          if (cachedDflash.arch) {
            const a = cachedDflash.arch;
            txt += ` — ${a.numLayers} layers, ${a.numKVHeads} KV heads × ${a.headDim}`;
          }
          info.innerHTML = txt;
        }
      }
    }

    // Show/hide mmproj section
    const mmprojSection = document.getElementById('mmproj-section');
    if (mmprojSection) {
      mmprojSection.style.display = cachedMmproj ? 'block' : 'none';
      if (cachedMmproj) {
        const info = document.getElementById('mmproj-info');
        if (info) {
          const url = `${HF_BASE}/${repo}/blob/main/${cachedMmproj.path}`;
          let html = `<div><a href="${url}" target="_blank" class="sidecar-link">${cachedMmproj.fileName}</a> (${cachedMmproj.sizeGB.toFixed(2)} GB)</div>`;
          if (cachedMmproj.allVariants && cachedMmproj.allVariants.length > 1) {
            const others = cachedMmproj.allVariants
              .filter(v => v.name !== cachedMmproj.fileName)
              .map(v => `<a href="${HF_BASE}/${repo}/blob/main/${v.path}" target="_blank" class="sidecar-link">${v.name}</a> <span style="color: var(--text-light);">${v.sizeGB.toFixed(1)} GB</span>`)
              .join(' &middot; ');
            html += `<div style="font-size: 0.7rem; color: var(--text-light); margin-top: 0.2rem;">${others}</div>`;
          }
          info.innerHTML = html;
        }
      }
    }

    // Update context slider
    const slider = document.getElementById('ctx-slider');
    const ctxKey = cachedGGUF?.context_length;
    const nativeCtx = ctxKey ? parseInt(ctxKey) : 131072;
    slider.max = nativeCtx;
    const currentCtx = parseInt(slider.value);
    slider.value = wasCached ? Math.min(currentCtx, nativeCtx) : nativeCtx;
    document.getElementById('ctx-display').textContent = fmtCtx(parseInt(slider.value));

    check();

  } catch (e) {
    errorEl.textContent = e.message;
    errorEl.style.display = 'block';
  } finally {
    btn.textContent = 'Load Model Metadata';
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Check / render
// ---------------------------------------------------------------------------
function check() {
  const errorEl = document.getElementById('error');
  const metaEl = document.getElementById('meta');
  const resultsEl = document.getElementById('results');
  errorEl.style.display = 'none';
  metaEl.innerHTML = '';
  resultsEl.innerHTML = '';

  if (!cachedGGUF || !cachedFiles) {
    errorEl.textContent = 'Load model first (click "Load Model Metadata")';
    errorEl.style.display = 'block';
    return;
  }

  const repo = normalizeRepo(document.getElementById('repo').value);
  const vram = parseFloat(document.getElementById('vram').value);
  const ctx = parseInt(document.getElementById('ctx-slider').value);
  const ubatch = parseInt(document.getElementById('ubatch-slider').value);
  const kBpp = parseFloat(document.getElementById('k-quant').value);
  const vBpp = parseFloat(document.getElementById('v-quant').value);
  const mtpkBpp = parseFloat(document.getElementById('mtp-k-quant').value);
  const mtpvBpp = parseFloat(document.getElementById('mtp-v-quant').value);

  if (!vram || !ctx) {
    errorEl.textContent = 'Fill in VRAM and context';
    errorEl.style.display = 'block';
    return;
  }

  const ggufFiles = cachedFiles
    .filter(f =>
      f.path.endsWith('.gguf') &&
      !f.path.startsWith('mmproj') &&
      !f.path.startsWith('imatrix') &&
      !f.path.startsWith('dflash-') &&
      !f.path.startsWith('MTP/') &&
      !f.path.startsWith('mtp-')
    )
    .sort((a, b) => b.size - a.size);

  if (!ggufFiles.length) {
    errorEl.textContent = 'No GGUF files found in this repo.';
    errorEl.style.display = 'block';
    return;
  }

  const arch = cachedArch;

    // Embedded MTP (in the main GGUF): when the toggle is off, estimate as if
    // the MTP tensors were absent (llama.cpp skips them when spec decoding is
    // disabled — no draft KV, no extra rs_seq).
    const mtpEnabled = document.getElementById('mtp-toggle')?.checked ?? true;
    const effArch = (arch && arch.mtpLayers > 0 && !mtpEnabled)
      ? withMtpDisabled(arch)
      : arch;

    // Assemble DFlash input for the estimator (only if toggle is on)
    const dflashEnabled = document.getElementById('dflash-toggle').checked;
    let dflashInput = null;
    let dflashkBpp = undefined;
    let dflashvBpp = undefined;
    if (cachedDflash && dflashEnabled) {
      dflashInput = {
        sizeGB: cachedDflash.sizeGB,
        arch: cachedDflash.arch,
      };
      dflashkBpp = parseFloat(document.getElementById('dflash-k-quant').value);
      dflashvBpp = parseFloat(document.getElementById('dflash-v-quant').value);
    }

    // mmproj GPU cost: weight size + compute buffer (only when mode is "gpu")
    const mmprojMode = document.getElementById('mmproj-mode').value;
    let mmprojGpuGB = 0;
    let mmprojComputeGB = 0;
    if (cachedMmproj && mmprojMode === 'gpu') {
      mmprojComputeGB = calcMmprojCompute(cachedMmproj.clipArch);
      mmprojGpuGB = cachedMmproj.sizeGB + mmprojComputeGB;
    }

    // MTP sidecar (only if toggle is on; mutually exclusive with embedded MTP)
    const mtpSidecarEnabled = document.getElementById('mtp-toggle')?.checked ?? false;
    let mtpSidecarInput = null;
    if (cachedMtpSidecar && mtpSidecarEnabled && !(effArch && effArch.mtpLayers > 0)) {
      mtpSidecarInput = {
        sizeGB: cachedMtpSidecar.sizeGB,
        arch: cachedMtpSidecar.arch,
      };
    }

    const estimate = estimateVram({
      ggufFiles, ctx, ubatch, arch: effArch, kBpp, vBpp, mtpkBpp, mtpvBpp,
      dflash: dflashInput, dflashkBpp, dflashvBpp,
      mtpSidecar: mtpSidecarInput,
    });

    renderMeta(repo, vram, ctx, effArch, estimate, ubatch, mmprojGpuGB, mmprojComputeGB);
    renderResults(repo, estimate, vram, mmprojGpuGB);
  renderCommand(repo, ctx, ubatch, effArch);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderMeta(repo, vram, ctx, arch, est, ubatch, mmprojGpuGB = 0, mmprojComputeGB = 0) {
  const metaEl = document.getElementById('meta');
  const reserveGB = parseFloat(document.getElementById('reserve').value) / 1024;
  const usableGB = vram - reserveGB;

  const archName = cachedGGUF?.architecture || 'unknown';
  const kLabel = document.getElementById('k-quant').selectedOptions[0].textContent;
  const vLabel = document.getElementById('v-quant').selectedOptions[0].textContent;
  const { kv, rs, mtp, computeBuf } = est;
  const nRsSeq = arch && arch.mtpLayers ? 2 : 0;

  let metaLines = `<div><strong>${repo}</strong></div>`;
  metaLines += `<div>Arch: ${archName} | Context: ${fmtCtx(ctx)} | K: ${kLabel} | V: ${vLabel}</div>`;
  if (kv.sizeGB > 0) {
    let kvDetail = `K: ${kv.kGB.toFixed(2)}, V: ${kv.vGB.toFixed(2)}, ${kv.nAttnLayers} attn`;
    if (kv.nSWA > 0) kvDetail += `, ${kv.nSWA} SWA (win: ${arch.slidingWindow})`;
    if (kv.padPct > 0) kvDetail += `, CUDA pad: +${kv.padPct.toFixed(1)}%`;
    metaLines += `<div>KV cache: ${kv.sizeGB.toFixed(2)} GB (${kvDetail})</div>`;
  }
  if (rs.sizeGB > 0) {
    metaLines += `<div>Recurrent state: ${rs.sizeGB.toFixed(2)} GB (${rs.nRecurLayers} Mamba layers, ${nRsSeq} rs_seq)</div>`;
  }
  if (mtp.sizeGB > 0) {
    const mtpKLabel = document.getElementById('mtp-k-quant').selectedOptions[0].textContent.split(' ')[0];
    const mtpVLabel = document.getElementById('mtp-v-quant').selectedOptions[0].textContent.split(' ')[0];
    let mtpDetail = `K: ${mtp.kGB.toFixed(2)}, V: ${mtp.vGB.toFixed(2)}, ${arch.mtpLayers} layer, ${mtpKLabel}/${mtpVLabel}`;
    if (mtp.padPct > 0) mtpDetail += `, CUDA pad: +${mtp.padPct.toFixed(1)}%`;
    metaLines += `<div>MTP draft KV: ${mtp.sizeGB.toFixed(2)} GB (${mtpDetail})</div>`;
  }
  if (est.dflash && est.dflash.total > 0) {
    const d = est.dflash;
    const dflashKLabel = document.getElementById('dflash-k-quant').selectedOptions[0].textContent.split(' ')[0];
    const dflashVLabel = document.getElementById('dflash-v-quant').selectedOptions[0].textContent.split(' ')[0];
    let dflashDetail = `${d.sizeGB.toFixed(2)} GB weights`;
    if (d.kv && d.kv.sizeGB > 0) dflashDetail += ` + ${d.kv.sizeGB.toFixed(2)} GB KV (${dflashKLabel}/${dflashVLabel})`;
    else if (cachedDflash && cachedDflash.headerError) dflashDetail += ` + ? GB KV (estimate failed: ${cachedDflash.headerError})`;
    const dflashLink = cachedDflash?.path
      ? `<a href="${HF_BASE}/${repo}/blob/main/${cachedDflash.path}" target="_blank" class="sidecar-link">${d.fileName || 'sidecar'}</a>`
      : (d.fileName || 'sidecar');
    metaLines += `<div>DFlash draft: ${dflashDetail} (${dflashLink})</div>`;
  }
  if (est.mtpSidecar && est.mtpSidecar.total > 0) {
    const m = est.mtpSidecar;
    let detail = `${m.sizeGB.toFixed(2)} GB weights`;
    if (m.computeGB > 0.001) detail += ` + ${m.computeGB.toFixed(2)} GB compute`;
    const mtpLink = cachedMtpSidecar?.path
      ? `<a href="${HF_BASE}/${repo}/blob/main/${cachedMtpSidecar.path}" target="_blank" class="sidecar-link">${m.fileName || 'sidecar'}</a>`
      : (m.fileName || 'sidecar');
    metaLines += `<div>MTP draft: ${detail} (${mtpLink})</div>`;
  }
  if (cachedMmproj) {
    const mmprojMode = document.getElementById('mmproj-mode').value;
    const mmprojLink = cachedMmproj.path
      ? `<a href="${HF_BASE}/${repo}/blob/main/${cachedMmproj.path}" target="_blank" class="sidecar-link">${cachedMmproj.fileName}</a>`
      : cachedMmproj.fileName;
    if (mmprojMode === 'gpu') {
      let detail = `${cachedMmproj.sizeGB.toFixed(2)} GB weights`;
      if (mmprojComputeGB > 0.001) detail += ` + ${mmprojComputeGB.toFixed(2)} GB compute`;
      else if (cachedMmproj.headerError) detail += ` + ? GB compute (estimate failed: ${cachedMmproj.headerError})`;
      metaLines += `<div>mmproj (GPU): ${mmprojGpuGB.toFixed(2)} GB (${detail}, ${mmprojLink})</div>`;
    } else if (mmprojMode === 'cpu') {
      metaLines += `<div>mmproj (CPU): ${cachedMmproj.sizeGB.toFixed(2)} GB — 0 GB GPU (--no-mmproj-offload) (${mmprojLink})</div>`;
    }
  }
  metaLines += `<div>Compute buffer (est): ${computeBuf.toFixed(1)} GB</div>`;
  metaLines += `<div style="margin-top: 0.5rem; font-weight: 600;">VRAM: ${vram} GB - ${reserveGB.toFixed(2)} GB reserve = ${usableGB.toFixed(2)} GB usable</div>`;
  metaEl.innerHTML = metaLines;
}

function renderResults(repo, est, vram, mmprojGpuGB = 0) {
  const resultsEl = document.getElementById('results');
  const reserveGB = parseFloat(document.getElementById('reserve').value) / 1024;
  const usableGB = vram - reserveGB;
  const { quants, kv, rs, mtp, computeBuf } = est;

  const showMmprojCol = mmprojGpuGB > 0;
  const showMtpSidecarCol = !!(est.mtpSidecar && est.mtpSidecar.total > 0);
  let html = `<div class="result-header">${quants.length} quants found &middot; <a href="https://huggingface.co/${repo}/tree/main" target="_blank">browse files &rarr;</a></div>`;
  html += `<div class="table-container"><table><thead><tr><th>Quant</th><th>Weights</th><th>K</th><th>V</th><th>RS</th><th>MTP</th><th>DFlash</th>${showMtpSidecarCol ? '<th>MTP draft</th>' : ''}${showMmprojCol ? '<th>mmproj</th>' : ''}<th>Compute</th><th>Total</th><th>Status</th></tr></thead><tbody>`;

  const dflashGB = est.dflash && est.dflash.total ? est.dflash.total : 0;
  const mtpSidecarGB = showMtpSidecarCol ? est.mtpSidecar.total : 0;
  const nCols = 10 + (showMtpSidecarCol ? 1 : 0) + (showMmprojCol ? 1 : 0);

  let addedSep = false;
  for (const q of quants) {
    const total = q.total + mmprojGpuGB;
    const fits = total <= usableGB;
    const tight = total > usableGB * 0.9 && fits;
    const headroom = usableGB - total;

    let badgeClass, badgeText, statusClass;
    if (fits) {
      if (tight) {
        badgeClass = 'badge-warn'; badgeText = 'tight'; statusClass = 'fit-warn';
      } else {
        badgeClass = 'badge-yes'; badgeText = 'fits'; statusClass = 'fit-yes';
      }
    } else {
      badgeClass = 'badge-no'; badgeText = 'over'; statusClass = 'fit-no';
    }

    const isLowQ = /(?:(?:UD-)?Q[23]|IQ[23])/i.test(q.quant);
    if (isLowQ && !addedSep) {
      html += `<tr class="row-sep"><td colspan="${nCols}" class="sep-label">Noticeable degradation beyond this point</td></tr>`;
      addedSep = true;
    }

    html += `<tr>
      <td><span class="badge ${badgeClass}">${badgeText}</span> <a href="https://huggingface.co/${repo}/blob/main/${q.file}" target="_blank" class="table-link">${q.quant}</a></td>
      <td>${q.sizeGB.toFixed(1)} GB</td>
      <td>${kv.kGB.toFixed(1)}</td>
      <td>${kv.vGB.toFixed(1)}</td>
      <td>${rs.sizeGB.toFixed(1)}</td>
      <td>${mtp.sizeGB.toFixed(1)}</td>
      <td>${dflashGB.toFixed(1)}</td>
      ${showMtpSidecarCol ? `<td>${mtpSidecarGB.toFixed(1)}</td>` : ''}
      ${showMmprojCol ? `<td>${mmprojGpuGB.toFixed(1)}</td>` : ''}
      <td>${computeBuf.toFixed(1)}</td>
      <td style="font-weight: 600;">${total.toFixed(1)} GB</td>
      <td class="${statusClass}">
        ${fits
          ? `<span class="headroom">${headroom.toFixed(1)} GB free</span>`
          : `+${Math.abs(headroom).toFixed(1)} GB over`
        }
      </td>
    </tr>`;
  }

  html += `</tbody></table></div>`;
  resultsEl.innerHTML = html;
}

function renderCommand(repo, ctx, ubatch, arch) {
  const cmdContainer = document.getElementById('cmd-container');
  const kCacheType = quantLabel(document.getElementById('k-quant'));
  const vCacheType = quantLabel(document.getElementById('v-quant'));

  const mtp = (arch && arch.mtpLayers > 0) ? {
    kCacheType: quantLabel(document.getElementById('mtp-k-quant')),
    vCacheType: quantLabel(document.getElementById('mtp-v-quant')),
  } : null;

  let dflash = null;
  if (cachedDflash && document.getElementById('dflash-toggle').checked) {
    dflash = {
      fileName: cachedDflash.fileName,
      kCacheType: quantLabel(document.getElementById('dflash-k-quant')),
      vCacheType: quantLabel(document.getElementById('dflash-v-quant')),
    };
  }

  // MTP sidecar: only when toggled on and no embedded MTP (mutually exclusive)
  let mtpSidecar = null;
  if (cachedMtpSidecar && (document.getElementById('mtp-toggle')?.checked ?? false) && !(arch && arch.mtpLayers > 0)) {
    mtpSidecar = { fileName: cachedMtpSidecar.fileName };
  }

  // mmproj: build options based on the 3-state dropdown
  let mmproj = null;
  if (cachedMmproj) {
    const mode = document.getElementById('mmproj-mode').value;
    if (mode !== 'off') {
      mmproj = {
        fileName: cachedMmproj.fileName,
        cpuOnly: mode === 'cpu',
      };
    }
  }

  const cmd = buildCommand({ repo, ctx, ubatch, kCacheType, vCacheType, arch, mtp, dflash, mtpSidecar, mmproj });

  const cmdEscaped = cmd.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  cmdContainer.innerHTML = `
    <div class="cmd-label" id="cmd-label">Ready to run (Click box to copy)</div>
    <div class="cmd-box" id="cmd-box" title="Click to copy to clipboard"><code>${cmdEscaped}</code></div>
    <div style="font-size: 0.75rem; color: var(--text-light); margin-top: 0.5rem; font-style: italic;">For best results, take this as just a starting point and play around with it.</div>
  `;

  document.getElementById('cmd-box').addEventListener('click', function() {
    navigator.clipboard.writeText(cmd).then(() => {
      const label = document.getElementById('cmd-label');
      const originalText = 'Ready to run (Click box to copy)';
      label.innerText = 'Copied to clipboard! ✓';
      label.style.color = 'var(--success-text)';
      setTimeout(() => {
        label.innerText = originalText;
        label.style.color = 'var(--text-light)';
      }, 2000);
    });
  });
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
const GPU_PRESETS = [
  ['RTX PRO 6000', 96],
  ['H100', 80],
  ['RTX 5090', 32],
  ['RTX 4090 / 3090', 24],
  ['RTX 5080 / 4080', 16],
  ['RTX 5070 / 3060', 12],
  ['RTX 5060 / 4060', 8],
];

const MODEL_PRESETS = [
  ['Qwen3.8 27B', 'unsloth/Qwen3.8-27B-GGUF'],
  ['Qwen3.6 27B', 'unsloth/Qwen3.6-27B-MTP-GGUF'],
  ['Muse Glimmer 30B', 'unsloth/Muse-Glimmer-30B-GGUF'],
  ['Gemma4 31B', 'unsloth/gemma-4-31b-it-GGUF'],
  ['Gemma4 12B', 'unsloth/gemma-4-12b-it-GGUF'],
  ['Qwen3.5 9B', 'unsloth/Qwen3.5-9B-GGUF'],
  ['Gemma4 E4B', 'unsloth/gemma-4-E4B-it-GGUF'],
  ['Qwen3.5 4B', 'unsloth/Qwen3.5-4B-GGUF'],
];
const MODEL_PRESETS_MOE = [
  ['Qwen3.6 35B A3B', 'unsloth/Qwen3.6-35B-A3B-GGUF'],
  ['Gemma4 26B A4B', 'unsloth/gemma-4-26B-A4B-it-GGUF'],
];

// ---------------------------------------------------------------------------
// Preset button builders
// ---------------------------------------------------------------------------
function buildPresets() {
  const gpuContainer = document.getElementById('gpu-presets');
  for (const [label, vram] of GPU_PRESETS) {
    const btn = document.createElement('button');
    btn.className = 'preset-btn';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      document.getElementById('vram').value = vram;
      gpuContainer.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (cachedGGUF) check();
    });
    gpuContainer.appendChild(btn);
  }
  gpuContainer.children[3].classList.add('active');

  const modelContainer = document.getElementById('model-presets');
  const denseLabel = document.createElement('span');
  denseLabel.className = 'preset-label';
  denseLabel.textContent = 'Dense';
  modelContainer.appendChild(denseLabel);
  for (const [label, repo] of MODEL_PRESETS) {
    const btn = document.createElement('button');
    btn.className = 'preset-btn';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      document.getElementById('repo').value = repo;
      modelContainer.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadModel();
    });
    modelContainer.appendChild(btn);
  }
  const moeLabel = document.createElement('span');
  moeLabel.className = 'preset-label';
  moeLabel.textContent = 'Mixture of Experts';
  modelContainer.appendChild(moeLabel);
  for (const [label, repo] of MODEL_PRESETS_MOE) {
    const btn = document.createElement('button');
    btn.className = 'preset-btn';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      document.getElementById('repo').value = repo;
      modelContainer.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadModel();
    });
    modelContainer.appendChild(btn);
  }
  const initialRepo = document.getElementById('repo').value;
  const allPresets = [...MODEL_PRESETS, ...MODEL_PRESETS_MOE];
  for (let i = 0; i < allPresets.length; i++) {
    if (allPresets[i][1] === initialRepo) {
      const btnIdx = i < MODEL_PRESETS.length ? i + 1 : i + 2;
      modelContainer.children[btnIdx].classList.add('active');
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------
function wireEvents() {
  const ctxSlider = document.getElementById('ctx-slider');
  const ctxDisplay = document.getElementById('ctx-display');
  ctxSlider.addEventListener('input', () => {
    ctxDisplay.textContent = fmtCtx(parseInt(ctxSlider.value));
    check();
  });

  const ubatchSlider = document.getElementById('ubatch-slider');
  const ubatchDisplay = document.getElementById('ubatch-display');
  ubatchSlider.addEventListener('input', () => {
    ubatchDisplay.textContent = ubatchSlider.value;
    check();
  });

  document.getElementById('repo').addEventListener('keydown', e => {
    if (e.key === 'Enter') loadModel();
  });

  document.querySelectorAll('#vram').forEach(el => {
    el.addEventListener('change', () => {
      document.querySelectorAll('#gpu-presets .preset-btn').forEach(b => b.classList.remove('active'));
      check();
    });
    el.addEventListener('keydown', e => { if (e.key === 'Enter') check(); });
  });
  document.getElementById('reserve').addEventListener('change', check);
  document.getElementById('reserve').addEventListener('keydown', e => { if (e.key === 'Enter') check(); });
  document.querySelectorAll('#k-quant, #v-quant, #mtp-k-quant, #mtp-v-quant, #dflash-k-quant, #dflash-v-quant, #dflash-toggle, #mtp-toggle, #mmproj-mode').forEach(el => {
    el.addEventListener('change', check);
    el.addEventListener('keydown', e => { if (e.key === 'Enter') check(); });
  });
  document.getElementById('repo').addEventListener('input', () => {
    document.querySelectorAll('#model-presets .preset-btn').forEach(b => b.classList.remove('active'));
  });

  document.getElementById('load').addEventListener('click', loadModel);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
export function init() {
  buildPresets();
  wireEvents();
  loadModel();
}
