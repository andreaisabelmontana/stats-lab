// ============================================================
// stats-lab — 15 visual probability + statistics demos.
//
// Every demo follows the same three-step pattern:
//   1. read slider state through `n(id, default)` → always finite
//   2. compute samples / curve into a local buffer
//   3. render in a single `draw()` wrapped in try/catch
//
// `draw` is idempotent: on every call it resets the canvas transform,
// clears the canvas, then draws from scratch. This means resizes and
// rapid slider input can never compound state.
// ============================================================

// The numeric and statistics cores live in framework-free ES modules under
// ./stats/ and are unit-tested with node:test. This file is the DOM/canvas
// layer: it reads sliders, calls the cores, and renders.
import { TAU, safe, clamp } from './stats/numeric.js';
import {
  normalPdf as npdf, normalCdf as ncdf, zstar,
  binomialPmf, geometricPmf, negBinomialPmf, poissonPmf,
  continuousPdf, continuousMoments, tPdf, tCdf,
} from './stats/distributions.js';
import { gauss, expRV, unitSamplers } from './stats/random.js';
import { leastSquares } from './stats/regression.js';
import { confidenceInterval, covers, zTest, bayesDiagnostic } from './stats/inference.js';
import { infer as bnInfer } from './stats/bayesnet.js';
import { step as mkStep, stationary as mkStationary } from './stats/markov.js';

// ---------- DOM helper ---------------------------------------------------
function n(id, fallback) {
  const el = document.getElementById(id);
  const v = el ? +el.value : NaN;
  return Number.isFinite(v) ? v : fallback;
}

// ---------- canvas helpers -----------------------------------------------
const ACCENT = '#4338CA';
const ACCENT_S = 'rgba(67,56,202,0.20)';
const RULE = '#E5E5EA';
const INK  = '#15151A';
const INK_S = '#4B4B55';
const MUTED = '#8A8A92';
const GOOD = '#16A34A';
const WARN = '#F59E0B';
const BAD  = '#DC2626';

function fitCanvas(cv) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = cv.getBoundingClientRect();
  const cssW = Math.max(80, rect.width);
  const cssH = Math.max(80, parseInt(cv.getAttribute('height'), 10) || 280);
  cv.width  = Math.floor(cssW * dpr);
  cv.height = Math.floor(cssH * dpr);
  cv.style.height = cssH + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);     // single source of truth — resets every call
  ctx.font = '11px Inter, sans-serif';
  ctx.textBaseline = 'alphabetic';
  return { ctx, w: cssW, h: cssH };
}

function axes(ctx, w, h, m, opts = {}) {
  ctx.strokeStyle = RULE; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, h - m.b);
  ctx.lineTo(w - m.r, h - m.b);
  ctx.stroke();
  if (opts.x) { ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.fillText(opts.x, m.l + (w - m.l - m.r) / 2, h - 4); }
  if (opts.y) {
    ctx.save(); ctx.translate(10, m.t + (h - m.t - m.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.fillText(opts.y, 0, 0); ctx.restore();
  }
}

// Auto-y-scale that ignores Infinity / outliers — the cause of the "blank canvas" bug
// when a PDF blows up near a singularity.
function robustMax(ys) {
  const v = ys.filter(Number.isFinite);
  if (!v.length) return 1;
  v.sort((a, b) => a - b);
  const q = v[Math.floor(v.length * 0.98)] || v[v.length - 1];
  return Math.max(1e-9, q * 1.1);
}

function setText(id, t) { const el = document.getElementById(id); if (el) el.textContent = t; }

// Wrap each demo init so a single bad demo can't break the page.
function mount(name, fn) {
  try { fn(); } catch (e) { console.error(`[stats-lab] ${name} init failed`, e); }
}

// =============================================================
// 1) Probability / Venn
// =============================================================
mount('prob', () => {
  const cv  = document.getElementById('cv-prob');
  const pA  = document.getElementById('pr-a'),  pB = document.getElementById('pr-b'), pI = document.getElementById('pr-i');
  const av  = document.getElementById('pr-av'), bv = document.getElementById('pr-bv'), iv = document.getElementById('pr-iv');

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    let a = clamp(n('pr-a', 0.4), 0.05, 0.95);
    let b = clamp(n('pr-b', 0.3), 0.05, 0.95);
    let i = clamp(n('pr-i', 0.1), 0, Math.min(a, b));
    // Keep the slider visually accurate
    pI.max = Math.min(a, b).toFixed(2);
    if (+pI.value > +pI.max) pI.value = pI.max;
    i = +pI.value;
    av.textContent = a.toFixed(2); bv.textContent = b.toFixed(2); iv.textContent = i.toFixed(2);

    const union = a + b - i;
    const cab = b > 0 ? i / b : 0;
    const cba = a > 0 ? i / a : 0;
    const indep = Math.abs(i - a * b) < 0.01;
    setText('pr-u', union.toFixed(3));
    setText('pr-cab', cab.toFixed(3));
    setText('pr-cba', cba.toFixed(3));
    setText('pr-ind', indep ? 'yes (approx.)' : 'no');

    // Draw two overlapping circles whose areas are proportional to P(A), P(B)
    // and whose overlap area ≈ P(A∩B). We use a simple lens-area solver.
    const cy = h / 2;
    const TOTAL = Math.min(w, h * 1.8) * 0.42;     // scale area budget
    const rA = Math.sqrt(a) * TOTAL;
    const rB = Math.sqrt(b) * TOTAL;
    const overlap = i;
    // Find d (distance between centres) such that lens area / (π·rA·rB area) ≈ overlap
    let d;
    if (overlap <= 0)                d = rA + rB;
    else if (overlap >= Math.min(a, b))  d = Math.abs(rA - rB) + 0.5;
    else {
      // numeric bisection on overlap area
      const target = Math.PI * TOTAL * TOTAL * overlap;
      let lo = Math.abs(rA - rB), hi = rA + rB;
      for (let k = 0; k < 30; k++) {
        const mid = (lo + hi) / 2;
        const aL = lensArea(rA, rB, mid);
        if (aL > target) lo = mid; else hi = mid;
      }
      d = (lo + hi) / 2;
    }

    const cxA = w / 2 - d / 2;
    const cxB = w / 2 + d / 2;

    // Universe rectangle (P = 1)
    ctx.strokeStyle = RULE; ctx.lineWidth = 1;
    ctx.strokeRect(12.5, 12.5, w - 25, h - 25);
    ctx.fillStyle = MUTED; ctx.textAlign = 'left';
    ctx.fillText('universe (P = 1)', 18, 24);

    ctx.fillStyle = 'rgba(67,56,202,0.25)';
    ctx.beginPath(); ctx.arc(cxA, cy, rA, 0, TAU); ctx.fill();
    ctx.fillStyle = 'rgba(220,38,38,0.25)';
    ctx.beginPath(); ctx.arc(cxB, cy, rB, 0, TAU); ctx.fill();
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(cxA, cy, rA, 0, TAU); ctx.stroke();
    ctx.strokeStyle = BAD;
    ctx.beginPath(); ctx.arc(cxB, cy, rB, 0, TAU); ctx.stroke();

    ctx.fillStyle = INK; ctx.textAlign = 'center'; ctx.font = '13px Inter';
    ctx.fillText('A', cxA - rA + 16, cy - rA + 16);
    ctx.fillText('B', cxB + rB - 16, cy - rB + 16);
  }
  // Closed-form circle-circle lens area
  function lensArea(r1, r2, d) {
    if (d >= r1 + r2) return 0;
    if (d <= Math.abs(r1 - r2)) return Math.PI * Math.min(r1, r2) ** 2;
    const part = (R, r, D) => R*R * Math.acos((D*D + R*R - r*r) / (2*D*R));
    const tri  = 0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
    return part(r1, r2, d) + part(r2, r1, d) - tri;
  }
  for (const el of [pA, pB, pI]) el.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 2) Discrete distributions
// =============================================================
mount('disc', () => {
  const cv = document.getElementById('cv-disc');
  const fE = document.getElementById('disc-f');
  const p1 = document.getElementById('disc-p1'), p1v = document.getElementById('disc-p1v'), p1L = document.getElementById('disc-p1-lbl');
  const p2 = document.getElementById('disc-p2'), p2v = document.getElementById('disc-p2v'), p2L = document.getElementById('disc-p2-lbl');

  // Per-family slider configuration. Keeps a slider in sync with a family
  // without ever letting a stale value leak from the previous family.
  const CFG = {
    bernoulli: { p1: { hidden: true,  v: 0 },                                 p2: { label: 'p',  min: 0.01, max: 0.99, step: 0.01, v: 0.5 } },
    binomial:  { p1: { label: 'n',    min: 1,    max: 100, step: 1,    v: 20 },p2: { label: 'p',  min: 0.01, max: 0.99, step: 0.01, v: 0.5 } },
    geometric: { p1: { hidden: true,  v: 0 },                                 p2: { label: 'p',  min: 0.05, max: 0.95, step: 0.01, v: 0.3 } },
    negbin:    { p1: { label: 'r',    min: 1,    max: 30,  step: 1,    v: 5  },p2: { label: 'p',  min: 0.05, max: 0.95, step: 0.01, v: 0.4 } },
    poisson:   { p1: { label: 'λ',    min: 0.1,  max: 30,  step: 0.1,  v: 4  },p2: { hidden: true, v: 0 } },
  };

  function syncSliders(fam) {
    for (const [el, lbl, lblWrap, cfg] of [
      [p1, document.querySelector('#disc-p1-lbl'), p1L, CFG[fam].p1],
      [p2, document.querySelector('#disc-p2-lbl'), p2L, CFG[fam].p2],
    ]) {
      if (cfg.hidden) { lbl.style.display = 'none'; continue; }
      lbl.style.display = '';
      el.min = cfg.min; el.max = cfg.max; el.step = cfg.step; el.value = cfg.v;
      // Update the label text (the leading text node)
      lbl.childNodes[0].nodeValue = cfg.label + ' ';
    }
  }
  syncSliders(fE.value);

  function pmf(fam, p1v, p2v) {
    const xs = [], ys = []; let mu = 0, vr = 0, mode = 0, modeP = -1;
    function push(x, p) { xs.push(x); ys.push(p); if (p > modeP) { modeP = p; mode = x; } mu += x * p; }
    if (fam === 'bernoulli') {
      const p = clamp(p2v, 1e-6, 1 - 1e-6);
      push(0, 1 - p); push(1, p); mu = p; vr = p * (1 - p);
    } else if (fam === 'binomial') {
      const nn = Math.max(1, Math.round(p1v));
      const p  = clamp(p2v, 1e-6, 1 - 1e-6);
      for (let k = 0; k <= nn; k++) push(k, binomialPmf(k, nn, p));
      mu = nn * p; vr = nn * p * (1 - p);
    } else if (fam === 'geometric') {
      const p = clamp(p2v, 1e-6, 1 - 1e-6);
      const k_max = Math.min(50, Math.ceil(5 / p));
      for (let k = 1; k <= k_max; k++) push(k, geometricPmf(k, p));
      mu = 1 / p; vr = (1 - p) / (p * p);
    } else if (fam === 'negbin') {
      const r = Math.max(1, Math.round(p1v));
      const p = clamp(p2v, 1e-6, 1 - 1e-6);
      const k_max = Math.min(80, Math.ceil((r * (1 - p)) / p * 4 + 10));
      for (let k = 0; k <= k_max; k++) push(k, negBinomialPmf(k, r, p));
      mu = r * (1 - p) / p; vr = r * (1 - p) / (p * p);
    } else if (fam === 'poisson') {
      const lam = Math.max(0.01, p1v);
      const k_max = Math.max(20, Math.ceil(lam + 4 * Math.sqrt(lam)));
      for (let k = 0; k <= k_max; k++) push(k, poissonPmf(k, lam));
      mu = lam; vr = lam;
    }
    return { xs, ys, mu, vr, mode };
  }

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const fam = fE.value;
    const p1Cfg = CFG[fam].p1, p2Cfg = CFG[fam].p2;
    const a = p1Cfg.hidden ? 0 : n('disc-p1', p1Cfg.v ?? 0);
    const b = p2Cfg.hidden ? 0 : n('disc-p2', p2Cfg.v ?? 0);
    p1v.textContent = p1Cfg.hidden ? '—' : (p1Cfg.step >= 1 ? Math.round(a) : a.toFixed(2));
    p2v.textContent = p2Cfg.hidden ? '—' : (p2Cfg.step >= 1 ? Math.round(b) : b.toFixed(2));

    const { xs, ys, mu, vr, mode } = pmf(fam, a, b);
    setText('disc-mean', mu.toFixed(3));
    setText('disc-var',  vr.toFixed(3));
    setText('disc-mode', mode.toString());

    const m = { l: 36, r: 12, t: 16, b: 26 };
    axes(ctx, w, h, m, { x: 'k' });
    const ymax = robustMax(ys);
    const bw = (w - m.l - m.r) / Math.max(1, xs.length);
    for (let i = 0; i < xs.length; i++) {
      const y = safe(ys[i], 0);
      const px = m.l + (i + 0.5) * bw;
      const py = h - m.b - (y / ymax) * (h - m.t - m.b);
      const isMode = xs[i] === mode;
      ctx.fillStyle = isMode ? ACCENT : ACCENT_S;
      ctx.fillRect(px - bw * 0.4, py, bw * 0.8, h - m.b - py);
      // x ticks (sparse)
      if (xs.length <= 30 || i % Math.ceil(xs.length / 12) === 0) {
        ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.font = '10px JetBrains Mono';
        ctx.fillText(xs[i], px, h - m.b + 12);
      }
    }
    // E[X] marker
    if (Number.isFinite(mu)) {
      const ex = m.l + ((mu - (xs[0] ?? 0)) / Math.max(1, xs.length - 1)) * (w - m.l - m.r) + bw / 2;
      ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
      ctx.moveTo(ex, m.t); ctx.lineTo(ex, h - m.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = INK; ctx.textAlign = 'left'; ctx.font = '10px JetBrains Mono';
      ctx.fillText(`E[X] = ${mu.toFixed(2)}`, ex + 4, m.t + 12);
    }
  }
  fE.addEventListener('change', () => { syncSliders(fE.value); draw(); });
  for (const el of [p1, p2]) el.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 3) Continuous distributions
// =============================================================
mount('cont', () => {
  const cv = document.getElementById('cv-cont');
  const fE = document.getElementById('cont-f');
  const p1 = document.getElementById('cont-p1'), p1v = document.getElementById('cont-p1v'), p1L = document.getElementById('cont-p1-lbl');
  const p2 = document.getElementById('cont-p2'), p2v = document.getElementById('cont-p2v'), p2L = document.getElementById('cont-p2-lbl');
  const cdfE = document.getElementById('cont-cdf');

  const CFG = {
    uniform:     { p1: { label: 'a',  min: -4, max: 2,  step: 0.05, v: 0  }, p2: { label: 'b', min: -2, max: 6,  step: 0.05, v: 1  } },
    normal:      { p1: { label: 'μ',  min: -3, max: 3,  step: 0.05, v: 0  }, p2: { label: 'σ', min: 0.1, max: 3,  step: 0.05, v: 1  } },
    exponential: { p1: { label: 'λ',  min: 0.1, max: 5, step: 0.05, v: 1  }, p2: { hidden: true } },
    gamma:       { p1: { label: 'k',  min: 0.5, max: 12, step: 0.1, v: 2  }, p2: { label: 'θ', min: 0.2, max: 5, step: 0.05, v: 1  } },
    beta:        { p1: { label: 'α',  min: 0.5, max: 10, step: 0.1, v: 2  }, p2: { label: 'β', min: 0.5, max: 10, step: 0.1, v: 5  } },
    chi2:        { p1: { label: 'k',  min: 1,  max: 30,  step: 1,    v: 4  }, p2: { hidden: true } },
  };

  function syncSliders(fam) {
    for (const [el, lbl, cfg] of [
      [p1, document.querySelector('#cont-p1-lbl'), CFG[fam].p1],
      [p2, document.querySelector('#cont-p2-lbl'), CFG[fam].p2],
    ]) {
      if (cfg.hidden) { lbl.style.display = 'none'; continue; }
      lbl.style.display = '';
      el.min = cfg.min; el.max = cfg.max; el.step = cfg.step; el.value = cfg.v;
      lbl.childNodes[0].nodeValue = cfg.label + ' ';
    }
  }
  syncSliders(fE.value);

  function range(fam, a, b) {
    if (fam === 'uniform')     return [Math.min(a, b) - 0.5, Math.max(a, b) + 0.5];
    if (fam === 'normal')      return [a - 4 * b, a + 4 * b];
    if (fam === 'exponential') return [0, 7 / Math.max(0.01, a)];
    if (fam === 'gamma')       return [0, a * b * 4 + 4 * Math.sqrt(a) * b];
    if (fam === 'beta')        return [0, 1];
    if (fam === 'chi2')        return [0, a + 4 * Math.sqrt(2 * a)];
  }
  const pdf = continuousPdf;       // (fam, x, a, b) — from ./stats/distributions.js
  const moments = continuousMoments; // (fam, a, b)

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const fam = fE.value;
    const p1Cfg = CFG[fam].p1, p2Cfg = CFG[fam].p2;
    const a = p1Cfg.hidden ? 0 : n('cont-p1', p1Cfg.v ?? 0);
    const b = p2Cfg.hidden ? 0 : n('cont-p2', p2Cfg.v ?? 0);
    p1v.textContent = p1Cfg.hidden ? '—' : (p1Cfg.step >= 1 ? Math.round(a) : a.toFixed(2));
    p2v.textContent = p2Cfg.hidden ? '—' : (p2Cfg.step >= 1 ? Math.round(b) : b.toFixed(2));

    const [lo, hi] = range(fam, a, b);
    const SAMPLES = 400;
    const xs = []; const ys = []; const ycdf = [];
    let cum = 0; const dx = (hi - lo) / SAMPLES;
    for (let i = 0; i <= SAMPLES; i++) {
      const x = lo + dx * i;
      const y = safe(pdf(fam, x, a, b), 0);
      xs.push(x); ys.push(y);
      cum += y * dx; ycdf.push(Math.min(1, cum));
    }
    const ymax = robustMax(ys);
    const M = moments(fam, a, b);
    setText('cont-mean', Number.isFinite(M.mu) ? M.mu.toFixed(3) : '∞');
    setText('cont-var',  Number.isFinite(M.vr) ? M.vr.toFixed(3) : '∞');

    const m = { l: 36, r: 12, t: 16, b: 26 };
    axes(ctx, w, h, m, { x: 'x' });
    const ax = x => m.l + (x - lo) / Math.max(1e-9, hi - lo) * (w - m.l - m.r);
    const ay = y => h - m.b - (y / ymax) * (h - m.t - m.b);

    // PDF fill
    ctx.fillStyle = ACCENT_S;
    ctx.beginPath();
    ctx.moveTo(ax(xs[0]), ay(0));
    for (let i = 0; i < xs.length; i++) ctx.lineTo(ax(xs[i]), ay(ys[i]));
    ctx.lineTo(ax(xs[xs.length - 1]), ay(0));
    ctx.closePath(); ctx.fill();
    // PDF stroke
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) i ? ctx.lineTo(ax(xs[i]), ay(ys[i])) : ctx.moveTo(ax(xs[i]), ay(ys[i]));
    ctx.stroke();

    // E[X] line
    if (Number.isFinite(M.mu) && M.mu >= lo && M.mu <= hi) {
      ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
      ctx.moveTo(ax(M.mu), m.t); ctx.lineTo(ax(M.mu), h - m.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = INK; ctx.textAlign = 'left'; ctx.font = '10px JetBrains Mono';
      ctx.fillText(`E[X] = ${M.mu.toFixed(2)}`, ax(M.mu) + 4, m.t + 12);
    }
    // CDF overlay
    if (cdfE.checked) {
      const cay = y => h - m.b - y * (h - m.t - m.b);
      ctx.strokeStyle = INK; ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) i ? ctx.lineTo(ax(xs[i]), cay(ycdf[i])) : ctx.moveTo(ax(xs[i]), cay(ycdf[i]));
      ctx.stroke();
      ctx.fillStyle = INK_S; ctx.textAlign = 'right';
      ctx.fillText('CDF', w - m.r - 4, m.t + 12);
    }
  }
  fE.addEventListener('change', () => { syncSliders(fE.value); draw(); });
  for (const el of [p1, p2, cdfE]) el.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 4) Convolution — density of X+Y by Monte Carlo
// =============================================================
mount('sum', () => {
  const cv = document.getElementById('cv-sum');
  const xE = document.getElementById('sum-x');
  const yE = document.getElementById('sum-y');
  const go = document.getElementById('sum-go');

  const SAMPLERS = {
    uniform: () => unitSamplers.uniform(),
    normal:  () => unitSamplers.normal(),
    exp:     () => unitSamplers.exp(),
    tri:     () => unitSamplers.tri(),
  };
  const RANGES = { uniform: [0, 1], normal: [-3, 3], exp: [0, 5], tri: [-1, 1] };

  function rangeFor(kx, ky) {
    const [a, b] = RANGES[kx], [c, d] = RANGES[ky];
    return [a + c, b + d];
  }

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const kx = xE.value, ky = yE.value;
    const [lo, hi] = rangeFor(kx, ky);
    const bins = 60, N = 20000;
    const hist = new Float64Array(bins);
    for (let i = 0; i < N; i++) {
      const s = SAMPLERS[kx]() + SAMPLERS[ky]();
      if (s < lo || s > hi) continue;
      const k = Math.min(bins - 1, Math.floor((s - lo) / (hi - lo) * bins));
      hist[k]++;
    }
    const cmax = robustMax(Array.from(hist));
    const m = { l: 30, r: 12, t: 16, b: 24 };
    axes(ctx, w, h, m, { x: 'x + y' });
    const bw = (w - m.l - m.r) / bins;
    for (let i = 0; i < bins; i++) {
      const x = m.l + i * bw;
      const y = h - m.b - (hist[i] / cmax) * (h - m.t - m.b);
      ctx.fillStyle = ACCENT_S; ctx.fillRect(x, y, bw - 0.5, h - m.b - y);
    }
    // X-axis ticks
    ctx.fillStyle = MUTED; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center';
    for (let k = 0; k <= 5; k++) {
      const xv = lo + (hi - lo) * k / 5;
      ctx.fillText(xv.toFixed(1), m.l + (k / 5) * (w - m.l - m.r), h - m.b + 12);
    }
    ctx.fillStyle = INK; ctx.textAlign = 'left';
    ctx.fillText(`${kx} + ${ky}`, m.l + 4, m.t + 12);
  }
  for (const el of [xE, yE, go]) el.addEventListener('input', draw);
  go.addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 5) Central Limit Theorem
// =============================================================
mount('clt', () => {
  const cv = document.getElementById('cv-clt');
  const src = document.getElementById('clt-src');
  const nE = document.getElementById('clt-n'),  nV = document.getElementById('clt-nv');
  const mE = document.getElementById('clt-m'),  mV = document.getElementById('clt-mv');
  const go = document.getElementById('clt-go');

  const SAMPLE = {
    uniform: () => Math.random(),
    exp:     () => -Math.log(Math.max(1e-12, Math.random())),
    bimodal: () => (Math.random() < 0.5 ? -2 : 2) + 0.5 * gauss(),
    cauchy:  () => Math.tan(Math.PI * (Math.random() - 0.5)),
  };
  const MOMENTS = {
    uniform: { mu: 0.5, sd: Math.sqrt(1 / 12) },
    exp:     { mu: 1,   sd: 1 },
    bimodal: { mu: 0,   sd: Math.sqrt(4 + 0.25) },
    cauchy:  null,
  };

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 32, r: 12, t: 16, b: 24 };
    const nn = Math.max(1, Math.round(n('clt-n', 10)));
    const M  = Math.max(100, Math.round(n('clt-m', 2000)));
    nV.textContent = nn; mV.textContent = M;
    const kind = src.value;

    const means = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      let s = 0;
      for (let j = 0; j < nn; j++) s += SAMPLE[kind]();
      means[i] = s / nn;
    }
    // Use 1–99 percentile range to avoid Cauchy's tails wrecking the axis
    const sorted = Array.from(means).filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length) return;
    const lo = sorted[Math.floor(sorted.length * 0.01)];
    const hi = sorted[Math.floor(sorted.length * 0.99)];
    const span = Math.max(1e-6, hi - lo);
    const bins = 40;
    const counts = new Float64Array(bins);
    for (const x of means) {
      if (!Number.isFinite(x) || x < lo || x > hi) continue;
      const k = Math.min(bins - 1, Math.floor((x - lo) / span * bins));
      counts[k]++;
    }
    const cmax = robustMax(Array.from(counts));
    const bw = (w - m.l - m.r) / bins;
    axes(ctx, w, h, m, { x: 'sample mean' });
    for (let i = 0; i < bins; i++) {
      const x = m.l + i * bw;
      const y = h - m.b - (counts[i] / cmax) * (h - m.t - m.b);
      ctx.fillStyle = ACCENT_S; ctx.fillRect(x, y, bw - 0.5, h - m.b - y);
    }
    const mom = MOMENTS[kind];
    if (mom) {
      const sd = mom.sd / Math.sqrt(nn);
      ctx.strokeStyle = INK; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const x = lo + span * i / 200;
        const expCount = M * npdf(x, mom.mu, sd) * (span / bins);
        const px = m.l + (x - lo) / span * (w - m.l - m.r);
        const py = h - m.b - (expCount / cmax) * (h - m.t - m.b);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = INK_S; ctx.textAlign = 'right'; ctx.font = '10px JetBrains Mono';
      ctx.fillText(`N(${mom.mu.toFixed(2)}, ${(mom.sd / Math.sqrt(nn)).toFixed(2)}²)`, w - m.r - 4, m.t + 12);
    } else {
      ctx.fillStyle = BAD; ctx.textAlign = 'right';
      ctx.fillText('Cauchy: no mean — CLT does not apply', w - m.r - 4, m.t + 12);
    }
  }
  for (const el of [src, nE, mE]) el.addEventListener('input', draw);
  go.addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 6) Bayes
// =============================================================
mount('bayes', () => {
  const cv = document.getElementById('cv-bayes');
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const P = clamp(n('bay-prior', 0.05), 0.001, 0.5);
    const S = clamp(n('bay-sens', 0.95), 0.5, 1);
    const T = clamp(n('bay-spec', 0.92), 0.5, 1);
    setText('bay-pv', P.toFixed(3));
    setText('bay-sv', S.toFixed(2));
    setText('bay-tv', T.toFixed(2));

    const N = 1000;
    const { tp: TP, fn: FN, fp: FP, tn: TN, posteriorPos: post, posteriorNeg: postn }
      = bayesDiagnostic(P, S, T, N);   // ./stats/inference.js
    setText('bay-post',  (post  * 100).toFixed(1) + '%');
    setText('bay-postn', (postn * 100).toFixed(2) + '%');

    const cols = 40, rows = 25, x0 = 8, y0 = 6;
    const cell = Math.min((w - 130) / cols, (h - 14) / rows);
    const grid = [];
    for (let i = 0; i < TN; i++) grid.push(0);
    for (let i = 0; i < TP; i++) grid.push(1);
    for (let i = 0; i < FP; i++) grid.push(2);
    for (let i = 0; i < FN; i++) grid.push(3);
    const COLOR = ['#E5E5EA', GOOD, WARN, BAD];
    for (let i = 0; i < N; i++) {
      const c = i % cols, r = Math.floor(i / cols);
      ctx.fillStyle = COLOR[grid[i]];
      ctx.fillRect(x0 + c * cell + 0.5, y0 + r * cell + 0.5, cell - 1.5, cell - 1.5);
    }
    const lx = x0 + cols * cell + 14, ly = y0 + 4;
    const legend = [
      ['true neg.',         COLOR[0], TN],
      ['caught (TP)',       GOOD,     TP],
      ['false alarm (FP)',  WARN,     FP],
      ['missed (FN)',       BAD,      FN],
    ];
    ctx.font = '11px Inter';
    legend.forEach((row, i) => {
      const y = ly + i * 22;
      ctx.fillStyle = row[1]; ctx.fillRect(lx, y, 10, 10);
      ctx.fillStyle = INK;   ctx.textAlign = 'left';
      ctx.fillText(row[0], lx + 16, y + 9);
      ctx.fillStyle = INK_S;
      ctx.fillText(`${row[2]} / ${N}`, lx + 16, y + 22);
    });
  }
  for (const id of ['bay-prior', 'bay-sens', 'bay-spec']) document.getElementById(id).addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 7) Maximum Likelihood
// =============================================================
mount('mle', () => {
  const cv = document.getElementById('cv-mle');
  const mE = document.getElementById('mle-m');
  const pE = document.getElementById('mle-p'), pv = document.getElementById('mle-pv');
  const re = document.getElementById('mle-resample');

  const MODELS = {
    'normal-mu':   { sample: () => gauss(0.7, 1),  range: [-3, 3], label: 'μ',
                     ll: (data, mu) => -0.5 * data.reduce((s, x) => s + (x - mu) ** 2, 0) - 0.5 * data.length * Math.log(2 * Math.PI) },
    'bernoulli-p': { sample: () => Math.random() < 0.35 ? 1 : 0, range: [0.01, 0.99], label: 'p',
                     ll: (data, p) => data.reduce((s, x) => s + (x ? Math.log(p) : Math.log(1 - p)), 0) },
    'exp-lam':     { sample: () => expRV(2),       range: [0.1, 5],  label: 'λ',
                     ll: (data, lam) => data.length * Math.log(lam) - lam * data.reduce((s, x) => s + x, 0) },
  };
  let data = [];
  function resample() {
    data = []; const M = MODELS[mE.value];
    for (let i = 0; i < 30; i++) data.push(M.sample());
  }
  resample();

  function setupSlider() {
    const M = MODELS[mE.value];
    pE.min = M.range[0]; pE.max = M.range[1]; pE.step = (M.range[1] - M.range[0]) / 200;
    pE.value = (M.range[0] + M.range[1]) / 2;
  }
  setupSlider();

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const M = MODELS[mE.value];
    const p = clamp(n('mle-p', (M.range[0] + M.range[1]) / 2), M.range[0], M.range[1]);
    pv.textContent = p.toFixed(3);

    // Sweep parameter
    const N = 200;
    const xs = [], ys = [];
    let best = -Infinity, bestP = M.range[0];
    for (let i = 0; i <= N; i++) {
      const pp = M.range[0] + (M.range[1] - M.range[0]) * i / N;
      const ll = M.ll(data, pp);
      xs.push(pp); ys.push(ll);
      if (ll > best) { best = ll; bestP = pp; }
    }
    const ymin = Math.min(...ys.filter(Number.isFinite)), ymax = Math.max(...ys.filter(Number.isFinite));
    if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) return;
    setText('mle-best', `${M.label} = ${bestP.toFixed(3)}`);
    setText('mle-ll',   M.ll(data, p).toFixed(2));

    const m = { l: 40, r: 12, t: 16, b: 26 };
    axes(ctx, w, h, m, { x: M.label, y: 'log L' });
    const ax = x => m.l + (x - M.range[0]) / (M.range[1] - M.range[0]) * (w - m.l - m.r);
    const ay = y => h - m.b - (y - ymin) / Math.max(1e-9, ymax - ymin) * (h - m.t - m.b);

    // log-likelihood curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) i ? ctx.lineTo(ax(xs[i]), ay(ys[i])) : ctx.moveTo(ax(xs[i]), ay(ys[i]));
    ctx.stroke();

    // MLE marker
    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(ax(bestP), m.t); ctx.lineTo(ax(bestP), h - m.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(ax(bestP), ay(best), 4, 0, TAU); ctx.fill();
    ctx.textAlign = 'center'; ctx.font = '10px JetBrains Mono';
    ctx.fillText('MLE', ax(bestP), m.t + 12);

    // Current parameter marker
    ctx.fillStyle = WARN; ctx.beginPath(); ctx.arc(ax(p), ay(M.ll(data, p)), 5, 0, TAU); ctx.fill();
  }
  mE.addEventListener('change', () => { resample(); setupSlider(); draw(); });
  pE.addEventListener('input', draw);
  re.addEventListener('click', () => { resample(); draw(); });
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 8) Confidence intervals
// =============================================================
mount('ci', () => {
  const cv = document.getElementById('cv-ci');
  const nE = document.getElementById('ci-n'), nV = document.getElementById('ci-nv');
  const lE = document.getElementById('ci-l'), lV = document.getElementById('ci-lv');
  const go = document.getElementById('ci-go');
  const cov = document.getElementById('ci-cov');

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 38, r: 12, t: 14, b: 24 };
    const nn = Math.max(2, Math.round(n('ci-n', 12)));
    const lev = clamp(n('ci-l', 0.95), 0.5, 0.999);
    nV.textContent = nn; lV.textContent = lev.toFixed(2);
    const reps = 100;
    let covered = 0;
    const intervals = [];
    for (let i = 0; i < reps; i++) {
      const sample = Array.from({ length: nn }, () => gauss());
      const ci = confidenceInterval(sample, lev);   // ./stats/inference.js
      intervals.push([ci.lo, ci.hi]);
      if (covers(ci, 0)) covered++;
    }
    cov.textContent = `${covered}/${reps}`;
    const lim = Math.max(2, 4 / Math.sqrt(nn));
    const ax = x => m.l + (x + lim) / (2 * lim) * (w - m.l - m.r);
    axes(ctx, w, h, m, { x: 'value' });
    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(ax(0), m.t); ctx.lineTo(ax(0), h - m.b); ctx.stroke();
    ctx.setLineDash([]);
    const slot = (h - m.t - m.b) / reps;
    for (let i = 0; i < reps; i++) {
      const y = m.t + slot * (i + 0.5);
      const [lo, hi] = intervals[i];
      const coversZero = lo <= 0 && hi >= 0;
      ctx.strokeStyle = coversZero ? ACCENT : BAD; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ax(lo), y); ctx.lineTo(ax(hi), y); ctx.stroke();
    }
  }
  for (const el of [nE, lE]) el.addEventListener('input', draw);
  go.addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 9) Hypothesis test + p-value
// =============================================================
mount('ht', () => {
  const cv = document.getElementById('cv-ht');
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 28, r: 12, t: 16, b: 26 };
    const delta = clamp(n('ht-d', 2), 0, 5);
    const thr   = clamp(n('ht-c', 1.64), -2, 6);
    const zobs  = clamp(n('ht-z', 2.1), -3, 6);
    setText('ht-dv', delta.toFixed(2));
    setText('ht-cv', thr.toFixed(2));
    setText('ht-zv', zobs.toFixed(2));

    const lo = Math.min(-4, -4 + delta - 2);
    const hi = Math.max(4, delta + 4);
    const ax = x => m.l + (x - lo) / (hi - lo) * (w - m.l - m.r);
    const ymax = 0.45;
    const ay = y => h - m.b - (y / ymax) * (h - m.t - m.b);

    const { alpha, beta, power, pValue: pval } = zTest(delta, thr, zobs); // ./stats/inference.js
    setText('ht-a',  alpha.toFixed(3));
    setText('ht-b',  beta.toFixed(3));
    setText('ht-p',  power.toFixed(3));
    setText('ht-pv', pval.toFixed(4));

    axes(ctx, w, h, m, { x: 'z' });

    function curveOutline(mu, stroke, fill) {
      const pts = [];
      for (let i = 0; i <= 200; i++) {
        const x = lo + (hi - lo) * i / 200;
        pts.push([x, npdf(x, mu, 1)]);
      }
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(ax(pts[0][0]), h - m.b);
      for (const [x, y] of pts) ctx.lineTo(ax(x), ay(y));
      ctx.lineTo(ax(pts[pts.length - 1][0]), h - m.b);
      ctx.fill();
      ctx.strokeStyle = stroke; ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach(([x, y], i) => i ? ctx.lineTo(ax(x), ay(y)) : ctx.moveTo(ax(x), ay(y)));
      ctx.stroke();
    }
    curveOutline(0,     BAD, 'rgba(220,38,38,0.05)');
    curveOutline(delta, ACCENT, 'rgba(67,56,202,0.05)');

    function shade(mu, from, to, color) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(ax(from), h - m.b);
      for (let i = 0; i <= 80; i++) {
        const x = from + (to - from) * i / 80;
        ctx.lineTo(ax(x), ay(npdf(x, mu, 1)));
      }
      ctx.lineTo(ax(to), h - m.b); ctx.closePath(); ctx.fill();
    }
    shade(0,     thr, hi,  'rgba(220,38,38,0.45)');   // α
    shade(delta, lo,  thr, 'rgba(245,158,11,0.55)');  // β
    shade(0,     zobs, hi, 'rgba(67,56,202,0.30)');   // p-value tail

    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(ax(thr), m.t); ctx.lineTo(ax(thr), h - m.b); ctx.stroke();
    ctx.setLineDash([]);
    // z_obs marker (solid)
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ax(zobs), m.t); ctx.lineTo(ax(zobs), h - m.b); ctx.stroke();

    ctx.fillStyle = INK; ctx.textAlign = 'center'; ctx.font = '11px Inter';
    ctx.fillText('H₀', ax(0),     m.t + 10);
    ctx.fillText('H₁', ax(delta), m.t + 10);
  }
  for (const id of ['ht-d', 'ht-c', 'ht-z']) document.getElementById(id).addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 10) t vs Normal
// =============================================================
mount('tn', () => {
  const cv = document.getElementById('cv-tn');
  const tpdf = tPdf;               // (x, df) — from ./stats/distributions.js
  const tcdfNumeric = tCdf;        // (c, df)

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const df = Math.max(1, Math.round(n('tn-d', 3)));
    const c  = clamp(n('tn-c', 2), 0, 5);
    setText('tn-dv', df);
    setText('tn-cv', c.toFixed(2));
    const tailT = 1 - tcdfNumeric(c, df);
    const tailN = 1 - ncdf(c);
    setText('tn-pt', tailT.toFixed(4));
    setText('tn-pn', tailN.toFixed(4));

    const m = { l: 32, r: 12, t: 14, b: 26 };
    const lo = -5, hi = 5;
    const ax = x => m.l + (x - lo) / (hi - lo) * (w - m.l - m.r);
    const ymax = 0.45;
    const ay = y => h - m.b - (y / ymax) * (h - m.t - m.b);
    axes(ctx, w, h, m, { x: 'x' });

    // tail shade (right of c, t distribution)
    ctx.fillStyle = 'rgba(67,56,202,0.30)';
    ctx.beginPath(); ctx.moveTo(ax(c), h - m.b);
    for (let i = 0; i <= 60; i++) {
      const x = c + (hi - c) * i / 60;
      ctx.lineTo(ax(x), ay(tpdf(x, df)));
    }
    ctx.lineTo(ax(hi), h - m.b); ctx.closePath(); ctx.fill();

    // t
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = lo + (hi - lo) * i / 200;
      const y = ay(tpdf(x, df));
      i ? ctx.lineTo(ax(x), y) : ctx.moveTo(ax(x), y);
    }
    ctx.stroke();
    // N(0,1)
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = lo + (hi - lo) * i / 200;
      const y = ay(npdf(x, 0, 1));
      i ? ctx.lineTo(ax(x), y) : ctx.moveTo(ax(x), y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // threshold
    ctx.strokeStyle = INK; ctx.beginPath(); ctx.moveTo(ax(c), m.t); ctx.lineTo(ax(c), h - m.b); ctx.stroke();

    ctx.fillStyle = ACCENT; ctx.textAlign = 'left'; ctx.font = '11px Inter';
    ctx.fillText(`t (df = ${df})`, m.l + 4, m.t + 12);
    ctx.fillStyle = INK;
    ctx.fillText('N(0, 1) — dashed', m.l + 4, m.t + 26);
  }
  for (const id of ['tn-d', 'tn-c']) document.getElementById(id).addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 11) Linear regression (draggable points)
// =============================================================
mount('reg', () => {
  const cv = document.getElementById('cv-reg');
  const b1 = document.getElementById('reg-b1');
  const b0 = document.getElementById('reg-b0');
  const r2 = document.getElementById('reg-r2');
  const rmseE = document.getElementById('reg-rmse');
  const res = document.getElementById('reg-res');
  const cb  = document.getElementById('reg-cb');
  const rst = document.getElementById('reg-reset');

  let pts = [];
  function resetPts() {
    pts = [];
    for (let i = 0; i < 15; i++) {
      const x = -3 + 6 * Math.random();
      const y = 0.6 * x + 0.5 + 0.7 * gauss();
      pts.push({ x, y });
    }
  }
  resetPts();

  const xmin = -4, xmax = 4, ymin = -4, ymax = 4;
  const m = { l: 32, r: 12, t: 14, b: 24 };
  let W = 0, H = 0;
  const ax = x => m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r);
  const ay = y => H - m.b - (y - ymin) / (ymax - ymin) * (H - m.t - m.b);
  const unax = px => xmin + (px - m.l) / (W - m.l - m.r) * (xmax - xmin);
  const unay = py => ymin + (H - m.b - py) / (H - m.t - m.b) * (ymax - ymin);
  let drag = null;

  // OLS fit via ./stats/regression.js; map field names the renderer expects.
  function fit() {
    const f = leastSquares(pts);
    return { slope: f.slope, intc: f.intercept, r2: f.r2, rmse: f.rmse, sxx: f.sxx, sse: f.sse, mx: f.meanX };
  }

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    W = w; H = h;
    ctx.clearRect(0, 0, w, h);
    axes(ctx, w, h, m, { x: 'x', y: 'y' });
    ctx.strokeStyle = '#F4F4F7';
    for (let v = Math.ceil(xmin); v <= xmax; v++) { ctx.beginPath(); ctx.moveTo(ax(v), m.t); ctx.lineTo(ax(v), h - m.b); ctx.stroke(); }
    for (let v = Math.ceil(ymin); v <= ymax; v++) { ctx.beginPath(); ctx.moveTo(m.l, ay(v)); ctx.lineTo(w - m.r, ay(v)); ctx.stroke(); }

    const F = fit();
    setText('reg-b1', F.slope.toFixed(3));
    setText('reg-b0', F.intc.toFixed(3));
    setText('reg-r2', F.r2.toFixed(3));
    setText('reg-rmse', F.rmse.toFixed(3));

    if (cb.checked && F.sxx > 0 && pts.length > 2) {
      // 95% confidence band for the regression LINE
      const sigma2 = F.sse / (pts.length - 2);
      const tcrit = 1.96;
      ctx.fillStyle = 'rgba(67,56,202,0.10)';
      ctx.beginPath();
      const NN = 60;
      for (let i = 0; i <= NN; i++) {
        const xv = xmin + (xmax - xmin) * i / NN;
        const se = Math.sqrt(sigma2 * (1 / pts.length + (xv - F.mx) ** 2 / F.sxx));
        const yv = F.slope * xv + F.intc + tcrit * se;
        i ? ctx.lineTo(ax(xv), ay(yv)) : ctx.moveTo(ax(xv), ay(yv));
      }
      for (let i = NN; i >= 0; i--) {
        const xv = xmin + (xmax - xmin) * i / NN;
        const se = Math.sqrt(sigma2 * (1 / pts.length + (xv - F.mx) ** 2 / F.sxx));
        const yv = F.slope * xv + F.intc - tcrit * se;
        ctx.lineTo(ax(xv), ay(yv));
      }
      ctx.closePath(); ctx.fill();
    }
    // OLS line
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ax(xmin), ay(F.slope * xmin + F.intc));
    ctx.lineTo(ax(xmax), ay(F.slope * xmax + F.intc));
    ctx.stroke();
    // Residuals
    if (res.checked) {
      ctx.strokeStyle = WARN; ctx.lineWidth = 1;
      for (const p of pts) {
        ctx.beginPath();
        ctx.moveTo(ax(p.x), ay(p.y));
        ctx.lineTo(ax(p.x), ay(F.slope * p.x + F.intc));
        ctx.stroke();
      }
    }
    // Points
    for (const p of pts) {
      ctx.fillStyle = INK;
      ctx.beginPath(); ctx.arc(ax(p.x), ay(p.y), 5, 0, TAU); ctx.fill();
    }
  }

  function pickAt(mx, my) {
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (Math.hypot(ax(p.x) - mx, ay(p.y) - my) <= 9) return i;
    }
    return -1;
  }
  cv.addEventListener('pointerdown', e => {
    const r = cv.getBoundingClientRect();
    const i = pickAt(e.clientX - r.left, e.clientY - r.top);
    if (i >= 0) drag = i;
    else { pts.push({ x: unax(e.clientX - r.left), y: unay(e.clientY - r.top) }); draw(); }
  });
  cv.addEventListener('pointermove', e => {
    if (drag === null) return;
    const r = cv.getBoundingClientRect();
    pts[drag].x = clamp(unax(e.clientX - r.left), xmin, xmax);
    pts[drag].y = clamp(unay(e.clientY - r.top), ymin, ymax);
    draw();
  });
  cv.addEventListener('pointerup',    () => { drag = null; });
  cv.addEventListener('pointerleave', () => { drag = null; });
  for (const el of [res, cb]) el.addEventListener('change', draw);
  rst.addEventListener('click', () => { resetPts(); draw(); });
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 12) Monte Carlo π
// =============================================================
mount('mc', () => {
  const cv = document.getElementById('cv-mc');
  const r  = document.getElementById('mc-r'), rv = document.getElementById('mc-rv');
  const ne = document.getElementById('mc-n');
  const ke = document.getElementById('mc-k');
  const pe = document.getElementById('mc-pi');
  const tg = document.getElementById('mc-toggle');
  const rs = document.getElementById('mc-reset');
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  let N = 0, K = 0, running = true, S = 0;
  let ctx = null;

  function initFrame() {
    const rect = cv.getBoundingClientRect();
    cv.width  = Math.floor(rect.width * dpr);
    cv.height = Math.floor(rect.width * dpr);
    cv.style.height = rect.width + 'px';
    ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    S = rect.width;
    ctx.fillStyle = '#FAFAFA'; ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = INK; ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
    ctx.beginPath(); ctx.moveTo(0, S); ctx.arc(0, S, S, -Math.PI / 2, 0); ctx.lineTo(0, S);
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.stroke();
  }
  initFrame();

  function throwSome() {
    if (!ctx) return;
    const rate = +r.value;
    const batch = Math.max(1, Math.round(rate / 30));
    rv.textContent = `${rate}/s`;
    for (let i = 0; i < batch; i++) {
      const x = Math.random(), y = Math.random();
      const inside = x * x + y * y <= 1;
      ctx.fillStyle = inside ? ACCENT : MUTED;
      ctx.fillRect(x * S, (1 - y) * S, 1.5, 1.5);
      N++; if (inside) K++;
    }
    ne.textContent = N;
    ke.textContent = K;
    pe.textContent = N ? (4 * K / N).toFixed(5) : '—';
  }
  const loop = () => { if (running) throwSome(); };
  setInterval(loop, 33);
  tg.addEventListener('click', () => { running = !running; tg.textContent = running ? 'pause' : 'resume'; });
  rs.addEventListener('click', () => { N = 0; K = 0; ne.textContent = 0; ke.textContent = 0; pe.textContent = '—'; initFrame(); });
  r.addEventListener('input', () => { rv.textContent = `${r.value}/s`; });
  window.addEventListener('resize', () => { initFrame(); N = 0; K = 0; ne.textContent = 0; ke.textContent = 0; pe.textContent = '—'; });
});

// =============================================================
// 13) Random walks
// =============================================================
mount('rw', () => {
  const cv = document.getElementById('cv-rw');
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 28, r: 12, t: 14, b: 24 };
    const K = Math.max(1, Math.round(n('rw-k', 40)));
    const B = clamp(n('rw-b', 0), -0.4, 0.4);
    setText('rw-kv', K);
    setText('rw-bv', B.toFixed(2));
    const T = 200;
    const lim = Math.max(8, Math.sqrt(T) * 2 + Math.abs(B * T));
    const ax = t => m.l + t / T * (w - m.l - m.r);
    const ay = y => m.t + (1 - (y + lim) / (2 * lim)) * (h - m.t - m.b);
    axes(ctx, w, h, m, { x: 't' });

    // envelope
    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let t = 0; t <= T; t++) { const y = ay(Math.sqrt(t) + B * t); t ? ctx.lineTo(ax(t), y) : ctx.moveTo(ax(t), y); }
    ctx.stroke();
    ctx.beginPath();
    for (let t = 0; t <= T; t++) { const y = ay(-Math.sqrt(t) + B * t); t ? ctx.lineTo(ax(t), y) : ctx.moveTo(ax(t), y); }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = RULE;
    ctx.beginPath(); ctx.moveTo(m.l, ay(0)); ctx.lineTo(w - m.r, ay(0)); ctx.stroke();

    ctx.lineWidth = 1.2;
    for (let i = 0; i < K; i++) {
      ctx.strokeStyle = `hsla(${(i * 53) % 360}, 70%, 45%, 0.45)`;
      ctx.beginPath();
      let pos = 0;
      ctx.moveTo(ax(0), ay(0));
      for (let t = 1; t <= T; t++) {
        pos += (Math.random() < 0.5 + B ? 1 : -1);
        ctx.lineTo(ax(t), ay(pos));
      }
      ctx.stroke();
    }
  }
  for (const id of ['rw-k', 'rw-b']) document.getElementById(id).addEventListener('input', draw);
  document.getElementById('rw-go').addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 7) Bayesian network — exact inference on the "wet grass" net,
//    with the explaining-away effect (Module II: Bayesian networks).
// =============================================================
mount('bn', () => {
  const cv = document.getElementById('cv-bn'); if (!cv) return;
  const sel = id => document.getElementById(id);
  const evOf = id => { const v = sel(id).value; return v === '' ? null : +v; };

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function edge(ctx, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    const ux = dx / L, uy = dy / L;
    const x0 = a.x + ux * 44, y0 = a.y + uy * 32;
    const x1 = b.x - ux * 46, y1 = b.y - uy * 34;
    ctx.strokeStyle = '#bcb29c'; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0), s = 8;
    ctx.fillStyle = '#bcb29c';
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - s * Math.cos(ang - 0.4), y1 - s * Math.sin(ang - 0.4));
    ctx.lineTo(x1 - s * Math.cos(ang + 0.4), y1 - s * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  }
  function node(ctx, c, label, p, ev, dead) {
    const w = 138, h = 58, x = c.x - w / 2, y = c.y - h / 2;
    let fill = '#fff', stroke = ACCENT;
    if (ev === 1) { fill = 'rgba(22,163,74,0.13)'; stroke = GOOD; }
    else if (ev === 0) { fill = 'rgba(220,38,38,0.11)'; stroke = BAD; }
    roundRect(ctx, x, y, w, h, 11);
    ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = ev == null ? 1.5 : 2.2; ctx.strokeStyle = stroke; ctx.stroke();
    ctx.textAlign = 'left';
    ctx.fillStyle = INK_S; ctx.font = '600 12px Inter, sans-serif';
    ctx.fillText(label, x + 13, y + 19);
    ctx.fillStyle = INK; ctx.font = '700 18px JetBrains Mono, monospace';
    ctx.fillText(dead ? '—' : (p * 100).toFixed(1) + '%', x + 13, y + 41);
    ctx.fillStyle = MUTED; ctx.font = '9.5px Inter, sans-serif';
    ctx.fillText(ev == null ? 'P(true)' : 'observed', x + 80, y + 40);
    ctx.fillStyle = RULE; ctx.fillRect(x + 13, y + 47, w - 26, 4);
    if (!dead) { ctx.fillStyle = ev === 1 ? GOOD : ev === 0 ? BAD : ACCENT; ctx.fillRect(x + 13, y + 47, (w - 26) * p, 4); }
  }

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const ev = { R: evOf('bn-r'), S: evOf('bn-s'), W: evOf('bn-w') };
    const post = bnInfer(ev);
    const dead = post.Z <= 0;            // impossible evidence (e.g. dry everything but wet grass)
    const R = { x: w * 0.5, y: 46 }, S = { x: w * 0.27, y: h - 54 }, W = { x: w * 0.73, y: h - 54 };
    edge(ctx, R, S); edge(ctx, R, W); edge(ctx, S, W);
    node(ctx, R, 'Rain', post.R || 0, ev.R, dead);
    node(ctx, S, 'Sprinkler', post.S || 0, ev.S, dead);
    node(ctx, W, 'Wet grass', post.W || 0, ev.W, dead);
    const pct = v => (dead ? 'n/a' : (v * 100).toFixed(1) + '%');
    setText('bn-pr', pct(post.R)); setText('bn-ps', pct(post.S)); setText('bn-pw', pct(post.W));
    if (dead) {
      ctx.fillStyle = BAD; ctx.font = '600 12px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('this evidence has probability 0 under the model', w / 2, h / 2 + 4);
    }
  }
  for (const id of ['bn-r', 'bn-s', 'bn-w']) sel(id).addEventListener('change', draw);
  sel('bn-reset').addEventListener('click', () => { for (const id of ['bn-r', 'bn-s', 'bn-w']) sel(id).value = ''; draw(); });
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
});

// =============================================================
// 15) Markov chain — distribution πP iterated to the stationary state,
//     with a 3-state diagram (Module III: Markov chains, sessions 8-9).
// =============================================================
mount('mk', () => {
  const cv = document.getElementById('cv-mk'); if (!cv) return;
  const PRESETS = {
    weather:  { names: ['Sunny', 'Cloudy', 'Rainy'], P: [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.45, 0.35]] },
    surfer:   { names: ['Page A', 'Page B', 'Page C'], P: [[0.1, 0.6, 0.3], [0.4, 0.1, 0.5], [0.5, 0.4, 0.1]] },
    triangle: { names: ['State 1', 'State 2', 'State 3'], P: [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]] },
  };
  let key = 'weather', pi = [1, 0, 0], t = 0, star = [1 / 3, 1 / 3, 1 / 3], timer = null;
  const get = id => document.getElementById(id);

  function rebuild() {
    key = get('mk-preset').value;
    star = mkStationary(PRESETS[key].P).dist;
    const names = PRESETS[key].names;
    [...get('mk-start').options].forEach((o, i) => { o.textContent = names[i]; });
    reset();
  }
  function reset() {
    if (timer) { clearInterval(timer); timer = null; get('mk-play').textContent = 'play'; }
    const s = +get('mk-start').value; pi = [0, 0, 0]; pi[s] = 1; t = 0; draw();
  }
  function stepOnce() { pi = mkStep(pi, PRESETS[key].P); t++; draw(); }

  function edge(ctx, A, B, p) {
    if (p < 0.02) return;
    const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, px = -uy, py = ux, off = 10;
    const x0 = A.x + ux * 30 + px * off, y0 = A.y + uy * 30 + py * off;
    const x1 = B.x - ux * 33 + px * off, y1 = B.y - uy * 33 + py * off;
    ctx.strokeStyle = '#cbbf9f'; ctx.lineWidth = 1 + p * 7;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const ang = Math.atan2(y1 - y0, x1 - x0), s = 7 + p * 3;
    ctx.fillStyle = '#cbbf9f';
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - s * Math.cos(ang - 0.45), y1 - s * Math.sin(ang - 0.45));
    ctx.lineTo(x1 - s * Math.cos(ang + 0.45), y1 - s * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fill();
    if (p >= 0.1) { ctx.fillStyle = MUTED; ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText(p.toFixed(2), (x0 + x1) / 2 + px * 7, (y0 + y1) / 2 + py * 7 + 3); ctx.textAlign = 'left'; }
  }
  function node(ctx, c, name, p, stay) {
    const r = 30;
    ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 7);
    ctx.fillStyle = `rgba(67,56,202,${(0.1 + 0.72 * p).toFixed(3)})`; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = ACCENT; ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillStyle = p > 0.45 ? '#fff' : INK; ctx.font = '700 14px JetBrains Mono, monospace';
    ctx.fillText((p * 100).toFixed(0) + '%', c.x, c.y + 5);
    ctx.fillStyle = INK_S; ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText(name, c.x, c.y - r - 7);
    ctx.fillStyle = MUTED; ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText('stay ' + stay.toFixed(2), c.x, c.y + r + 13);
    ctx.textAlign = 'left';
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const P = PRESETS[key].P, names = PRESETS[key].names;
    const N = [{ x: w * 0.5, y: 56 }, { x: w * 0.27, y: 196 }, { x: w * 0.73, y: 196 }];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (i !== j) edge(ctx, N[i], N[j], P[i][j]);
    for (let i = 0; i < 3; i++) node(ctx, N[i], names[i], pi[i], P[i][i]);

    // distribution bars (current) with stationary tick marks
    const bx = 46, bw = (w - 92) / 3, by1 = h - 22, bh = 78, by0 = by1 - bh;
    ctx.strokeStyle = RULE; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx - 6, by1); ctx.lineTo(w - 40, by1); ctx.stroke();
    for (let i = 0; i < 3; i++) {
      const x = bx + i * bw, bwid = bw * 0.5, cx = x + bw / 2 - bwid / 2;
      ctx.fillStyle = ACCENT_S; ctx.fillRect(cx, by1 - pi[i] * bh, bwid, pi[i] * bh);
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.2; ctx.strokeRect(cx, by1 - pi[i] * bh, bwid, pi[i] * bh);
      // stationary marker
      const sy = by1 - star[i] * bh;
      ctx.strokeStyle = BAD; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx - 5, sy); ctx.lineTo(cx + bwid + 5, sy); ctx.stroke();
      ctx.fillStyle = INK_S; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(names[i], x + bw / 2, by1 + 14);
      ctx.fillStyle = INK; ctx.font = '600 10px JetBrains Mono, monospace';
      ctx.fillText((pi[i] * 100).toFixed(1) + '%', x + bw / 2, by0 - 4);
      ctx.textAlign = 'left';
    }
    ctx.fillStyle = BAD; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'right';
    ctx.fillText('— stationary π*', w - 40, by0 - 4); ctx.textAlign = 'left';

    setText('mk-step', t);
    const drift = pi.reduce((s, p, i) => s + Math.abs(p - star[i]), 0);
    const conv = drift < 0.01;
    setText('mk-conv', conv ? 'yes — converged' : 'not yet');
    get('mk-conv').style.color = conv ? GOOD : WARN;
  }
  get('mk-preset').addEventListener('change', rebuild);
  get('mk-start').addEventListener('change', reset);
  get('mk-go').addEventListener('click', stepOnce);
  get('mk-play').addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; get('mk-play').textContent = 'play'; return; }
    get('mk-play').textContent = 'pause';
    timer = setInterval(() => {
      const drift = pi.reduce((s, p, i) => s + Math.abs(p - star[i]), 0);
      if (drift < 0.002) { clearInterval(timer); timer = null; get('mk-play').textContent = 'play'; return; }
      stepOnce();
    }, 650);
  });
  get('mk-reset').addEventListener('click', reset);
  window.addEventListener('resize', draw);
  rebuild();
});
