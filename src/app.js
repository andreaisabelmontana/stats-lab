// ============================================================
// stats-lab — eight self-contained demos.
// All canvas drawing is pure 2D; all randomness uses Math.random.
// ============================================================

const TAU = Math.PI * 2;

// ---------- math helpers --------------------------------------------------
function lcg() { return Math.random(); }                          // alias for clarity
function gauss(mu = 0, sd = 1) {                                  // Box–Muller normal sample
  const u = Math.max(1e-12, lcg()), v = lcg();
  return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
}
function logGamma(z) {
  // Lanczos approximation
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
             771.32342877765313, -176.61502916214059, 12.507343278686905,
             -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(TAU) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function fact(n) { return Math.round(Math.exp(logGamma(n + 1))); }
function logComb(n, k) { return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1); }
function erf(x) {
  // Abramowitz–Stegun approximation, max error ~1.5e-7
  const s = Math.sign(x); x = Math.abs(x);
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * Math.exp(-x*x);
  return s * y;
}
function normCDF(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
function normPDF(x, mu, sd) {
  const z = (x - mu) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(TAU));
}
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

// ---------- canvas helpers ------------------------------------------------
function setupCanvas(cv) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = cv.getBoundingClientRect();
  cv.width = Math.floor(rect.width * dpr);
  cv.height = Math.floor(cv.height ? cv.height : 280 * dpr);
  // Use the height the HTML attribute requested, but in device pixels.
  cv.height = Math.floor(parseInt(cv.getAttribute('height'), 10) * dpr);
  cv.style.height = parseInt(cv.getAttribute('height'), 10) + 'px';
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.font = '11px Inter, sans-serif';
  return { ctx, w: rect.width, h: parseInt(cv.getAttribute('height'), 10) };
}

const ACCENT = '#4338CA';
const ACCENT_SOFT = 'rgba(67,56,202,0.20)';
const RULE = '#E5E5EA';
const INK = '#15151A';
const INK_S = '#4B4B55';
const MUTED = '#8A8A92';
const GOOD = '#16A34A';
const WARN = '#F59E0B';
const BAD  = '#DC2626';

function axes(ctx, w, h, m, opts = {}) {
  ctx.strokeStyle = RULE; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(m.l, m.t);     ctx.lineTo(m.l, h - m.b);
  ctx.lineTo(w - m.r, h - m.b);
  ctx.stroke();
  if (opts.yLabel) {
    ctx.save();
    ctx.translate(8, m.t + (h - m.t - m.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = MUTED; ctx.textAlign = 'center';
    ctx.fillText(opts.yLabel, 0, 0);
    ctx.restore();
  }
  if (opts.xLabel) {
    ctx.fillStyle = MUTED; ctx.textAlign = 'center';
    ctx.fillText(opts.xLabel, m.l + (w - m.l - m.r) / 2, h - 4);
  }
}

// ============================================================
// 1) Distributions
// ============================================================
(function demo_dist() {
  const cv = document.getElementById('cv-dist');
  const fam = document.getElementById('dist-family');
  const p1  = document.getElementById('dist-p1');
  const p2  = document.getElementById('dist-p2');
  const p1v = document.getElementById('dist-p1v');
  const p2v = document.getElementById('dist-p2v');
  const cdf = document.getElementById('dist-cdf');

  const PARAMS = {
    normal:  { p1: { min: -3, max: 3, step: 0.05, val: 0,   label: 'μ' },
               p2: { min: 0.1, max: 3, step: 0.05, val: 1,   label: 'σ' } },
    binom:   { p1: { min: 1, max: 60, step: 1,    val: 20,  label: 'n' },
               p2: { min: 0.05, max: 0.95, step: 0.01, val: 0.5, label: 'p' } },
    poisson: { p1: { min: 0.1, max: 30, step: 0.1, val: 4,   label: 'λ' },
               p2: { min: 0, max: 1, step: 1, val: 0, label: '—' } },
    expo:    { p1: { min: 0.1, max: 5, step: 0.05, val: 1,   label: 'λ' },
               p2: { min: 0, max: 1, step: 1, val: 0, label: '—' } },
    beta:    { p1: { min: 0.3, max: 8, step: 0.1, val: 2,    label: 'α' },
               p2: { min: 0.3, max: 8, step: 0.1, val: 5,    label: 'β' } },
  };

  function syncParams(f) {
    const P = PARAMS[f];
    p1.min = P.p1.min; p1.max = P.p1.max; p1.step = P.p1.step; p1.value = P.p1.val;
    p2.min = P.p2.min; p2.max = P.p2.max; p2.step = P.p2.step; p2.value = P.p2.val;
    p1.disabled = false;
    p2.disabled = P.p2.label === '—';
  }
  syncParams(fam.value);

  function draw() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 36, r: 12, t: 14, b: 28 };
    const a = +p1.value, b = +p2.value;
    p1v.textContent = a.toFixed(2);
    p2v.textContent = b.toFixed(2);

    const f = fam.value;
    let xs, ys, cy, discrete = false, label = '';

    if (f === 'normal') {
      label = `μ=${a.toFixed(2)}, σ=${b.toFixed(2)}`;
      const lo = a - 4 * b, hi = a + 4 * b;
      xs = []; ys = [];
      for (let i = 0; i <= 200; i++) {
        const x = lo + (hi - lo) * i / 200;
        xs.push(x); ys.push(normPDF(x, a, b));
      }
      cy = xs.map(x => normCDF((x - a) / b));
    } else if (f === 'binom') {
      label = `n=${a.toFixed(0)}, p=${b.toFixed(2)}`;
      discrete = true;
      const n = Math.round(a);
      xs = []; ys = [];
      let cum = 0; cy = [];
      for (let k = 0; k <= n; k++) {
        const p = Math.exp(logComb(n, k) + k * Math.log(b) + (n - k) * Math.log(1 - b));
        xs.push(k); ys.push(p); cum += p; cy.push(cum);
      }
    } else if (f === 'poisson') {
      label = `λ=${a.toFixed(2)}`;
      discrete = true;
      const lam = a;
      const k_max = Math.max(20, Math.ceil(lam + 4 * Math.sqrt(lam)));
      xs = []; ys = []; cy = []; let cum = 0;
      for (let k = 0; k <= k_max; k++) {
        const p = Math.exp(-lam + k * Math.log(lam) - logGamma(k + 1));
        xs.push(k); ys.push(p); cum += p; cy.push(cum);
      }
    } else if (f === 'expo') {
      label = `λ=${a.toFixed(2)}`;
      const lam = a;
      const hi = 5 / lam;
      xs = []; ys = []; cy = [];
      for (let i = 0; i <= 200; i++) {
        const x = (hi * i) / 200;
        xs.push(x); ys.push(lam * Math.exp(-lam * x));
        cy.push(1 - Math.exp(-lam * x));
      }
    } else if (f === 'beta') {
      label = `α=${a.toFixed(2)}, β=${b.toFixed(2)}`;
      xs = []; ys = [];
      const norm = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b));
      for (let i = 1; i < 200; i++) {
        const x = i / 200;
        const v = norm * Math.pow(x, a - 1) * Math.pow(1 - x, b - 1);
        xs.push(x); ys.push(v);
      }
      // Numerical CDF
      let cum = 0; cy = ys.map(v => (cum += v / 200));
    }

    const xmin = xs[0], xmax = xs[xs.length - 1];
    const ymax = Math.max(...ys) * 1.15 || 1;
    const ax = x => m.l + (x - xmin) / (xmax - xmin) * (w - m.l - m.r);
    const ay = y => h - m.b - (y / ymax) * (h - m.t - m.b);

    axes(ctx, w, h, m, { xLabel: 'x' });

    // PDF/PMF
    if (discrete) {
      const bw = (w - m.l - m.r) / (xs.length + 1) * 0.8;
      ctx.fillStyle = ACCENT_SOFT;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1;
      for (let i = 0; i < xs.length; i++) {
        const x = ax(xs[i]), y = ay(ys[i]);
        ctx.fillRect(x - bw / 2, y, bw, h - m.b - y);
        ctx.strokeRect(x - bw / 2, y, bw, h - m.b - y);
      }
    } else {
      ctx.fillStyle = ACCENT_SOFT;
      ctx.beginPath();
      ctx.moveTo(ax(xs[0]), h - m.b);
      for (let i = 0; i < xs.length; i++) ctx.lineTo(ax(xs[i]), ay(ys[i]));
      ctx.lineTo(ax(xs[xs.length - 1]), h - m.b);
      ctx.fill();
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const x = ax(xs[i]), y = ay(ys[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    if (cdf.checked) {
      ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
      ctx.beginPath();
      const cmax = 1;
      const cy_to = y => h - m.b - (y / cmax) * (h - m.t - m.b);
      for (let i = 0; i < xs.length; i++) {
        const x = ax(xs[i]), y = cy_to(cy[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = MUTED; ctx.textAlign = 'right';
      ctx.fillText('CDF', w - m.r - 4, m.t + 12);
    }

    ctx.fillStyle = INK_S; ctx.textAlign = 'left';
    ctx.fillText(label, m.l + 4, m.t + 12);
  }

  fam.addEventListener('change', () => { syncParams(fam.value); draw(); });
  for (const el of [p1, p2, cdf]) el.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();

// ============================================================
// 2) Central Limit Theorem
// ============================================================
(function demo_clt() {
  const cv = document.getElementById('cv-clt');
  const src = document.getElementById('clt-src');
  const nE  = document.getElementById('clt-n');
  const mE  = document.getElementById('clt-m');
  const nV  = document.getElementById('clt-nv');
  const mV  = document.getElementById('clt-mv');
  const go  = document.getElementById('clt-go');

  function sample(kind) {
    if (kind === 'uniform') return lcg();
    if (kind === 'exp')     return -Math.log(Math.max(1e-9, lcg()));
    if (kind === 'cauchy')  return Math.tan(Math.PI * (lcg() - 0.5));
    if (kind === 'bimodal') return (lcg() < 0.5 ? -2 : 2) + 0.5 * gauss();
  }
  function moments(kind) {
    if (kind === 'uniform') return { mu: 0.5, sd: Math.sqrt(1 / 12) };
    if (kind === 'exp')     return { mu: 1,   sd: 1 };
    if (kind === 'bimodal') return { mu: 0,   sd: Math.sqrt(4 + 0.25) };
    return null;                                        // Cauchy: no mean!
  }

  function draw() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 32, r: 12, t: 14, b: 24 };
    const n = +nE.value, M = +mE.value;
    nV.textContent = n; mV.textContent = M;
    const kind = src.value;

    // Collect M sample means of size n
    const means = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += sample(kind);
      means[i] = s / n;
    }
    // Use 1–99 percentile range to clip Cauchy's extremes
    const sorted = Array.from(means).sort((a, b) => a - b);
    const lo = sorted[Math.floor(M * 0.01)] ?? -3;
    const hi = sorted[Math.floor(M * 0.99)] ?? 3;
    const bins = 40;
    const counts = new Float64Array(bins);
    for (const x of means) {
      if (x < lo || x > hi) continue;
      const k = Math.min(bins - 1, Math.floor((x - lo) / (hi - lo) * bins));
      counts[k]++;
    }
    const cmax = Math.max(...counts);
    const bw = (w - m.l - m.r) / bins;
    axes(ctx, w, h, m, { xLabel: 'sample mean' });

    ctx.fillStyle = ACCENT_SOFT; ctx.strokeStyle = ACCENT; ctx.lineWidth = 1;
    for (let i = 0; i < bins; i++) {
      const x = m.l + i * bw;
      const y = h - m.b - (counts[i] / cmax) * (h - m.t - m.b);
      ctx.fillRect(x, y, bw - 1, h - m.b - y);
    }

    // Overlay theoretical normal if defined
    const mom = moments(kind);
    if (mom) {
      const sd = mom.sd / Math.sqrt(n);
      ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = lo + (hi - lo) * i / 100;
        const pdf = normPDF(x, mom.mu, sd);
        // Convert PDF -> expected count: M * pdf * binWidth
        const expCount = M * pdf * ((hi - lo) / bins);
        const px = m.l + (x - lo) / (hi - lo) * (w - m.l - m.r);
        const py = h - m.b - (expCount / cmax) * (h - m.t - m.b);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = INK_S; ctx.textAlign = 'right';
      ctx.fillText(`N(${mom.mu.toFixed(2)}, σ²/n)`, w - m.r - 4, m.t + 12);
    } else {
      ctx.fillStyle = BAD; ctx.textAlign = 'right';
      ctx.fillText('Cauchy: no mean — CLT fails', w - m.r - 4, m.t + 12);
    }
  }
  for (const el of [src, nE, mE, go]) el.addEventListener('input', draw);
  go.addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();

// ============================================================
// 3) Bayes — 100-square grid
// ============================================================
(function demo_bayes() {
  const cv  = document.getElementById('cv-bayes');
  const pr  = document.getElementById('bay-prior');
  const sn  = document.getElementById('bay-sens');
  const sp  = document.getElementById('bay-spec');
  const pv  = document.getElementById('bay-pv'), sv = document.getElementById('bay-sv'), tv = document.getElementById('bay-tv');
  const post = document.getElementById('bay-post');

  function draw() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const P = +pr.value, S = +sn.value, T = +sp.value;
    pv.textContent = P.toFixed(3); sv.textContent = S.toFixed(2); tv.textContent = T.toFixed(2);

    const N = 1000;
    const D = Math.round(N * P);          // diseased
    const ND = N - D;
    const TP = Math.round(D * S);
    const FN = D - TP;
    const FP = Math.round(ND * (1 - T));
    const TN = ND - FP;
    const postP = (TP + FP) > 0 ? TP / (TP + FP) : 0;
    post.textContent = (postP * 100).toFixed(1) + '%';

    // Render 100 squares -> each square = 10 people
    const cols = 40, rows = 25;
    const cell = Math.min((w - 90) / cols, (h - 20) / rows);
    const x0 = 8, y0 = 10;

    const grid = []; // 0=TN, 1=TP, 2=FP, 3=FN
    for (let i = 0; i < TN; i++) grid.push(0);
    for (let i = 0; i < TP; i++) grid.push(1);
    for (let i = 0; i < FP; i++) grid.push(2);
    for (let i = 0; i < FN; i++) grid.push(3);
    // Keep order so colours form bands.
    const COLOR = ['#E5E5EA', GOOD, WARN, BAD];

    for (let i = 0; i < N; i++) {
      const c = i % cols, r = Math.floor(i / cols);
      ctx.fillStyle = COLOR[grid[i]];
      ctx.fillRect(x0 + c * cell + 0.5, y0 + r * cell + 0.5, cell - 1.5, cell - 1.5);
    }

    // Legend
    const lx = x0 + cols * cell + 14, ly = y0 + 4;
    const legend = [
      ['true neg.', COLOR[0], TN],
      ['caught (TP)', GOOD, TP],
      ['false alarm (FP)', WARN, FP],
      ['missed (FN)', BAD, FN],
    ];
    ctx.font = '11px Inter, sans-serif';
    legend.forEach((row, i) => {
      const y = ly + i * 22;
      ctx.fillStyle = row[1];
      ctx.fillRect(lx, y, 10, 10);
      ctx.fillStyle = INK; ctx.textAlign = 'left';
      ctx.fillText(row[0], lx + 16, y + 9);
      ctx.fillStyle = INK_S;
      ctx.fillText(`${row[2]} / ${N}`, lx + 16, y + 22);
    });
  }
  for (const el of [pr, sn, sp]) el.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();

// ============================================================
// 4) Confidence intervals
// ============================================================
(function demo_ci() {
  const cv = document.getElementById('cv-ci');
  const nE = document.getElementById('ci-n');
  const lE = document.getElementById('ci-l');
  const nV = document.getElementById('ci-nv');
  const lV = document.getElementById('ci-lv');
  const go = document.getElementById('ci-go');
  const cov = document.getElementById('ci-cov');

  function zStar(level) {
    // Inverse normal CDF for (1+level)/2 via Newton on erf
    let z = 1.96, target = (1 + level) / 2;
    for (let i = 0; i < 50; i++) {
      const f = normCDF(z) - target;
      const df = normPDF(z, 0, 1);
      z -= f / df;
    }
    return z;
  }

  function draw() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 38, r: 12, t: 14, b: 24 };
    const n = +nE.value, lev = +lE.value;
    nV.textContent = n; lV.textContent = lev.toFixed(2);

    const reps = 100;
    const z = zStar(lev);
    let covered = 0;

    // Generate intervals + draw them
    const intervals = [];
    for (let i = 0; i < reps; i++) {
      let sum = 0, sq = 0;
      for (let j = 0; j < n; j++) { const x = gauss(); sum += x; sq += x * x; }
      const mean = sum / n;
      const sd = Math.sqrt(Math.max(1e-9, (sq - n * mean * mean) / Math.max(1, n - 1)));
      const half = z * sd / Math.sqrt(n);
      intervals.push([mean - half, mean + half]);
      if (-0 >= mean - half && -0 <= mean + half) covered++;
    }
    cov.textContent = `${covered}/${reps}`;

    // Determine x range (rounded to ±3/√n)
    const lim = Math.max(2, 4 / Math.sqrt(n));
    const ax = x => m.l + (x + lim) / (2 * lim) * (w - m.l - m.r);
    axes(ctx, w, h, m, { xLabel: 'value' });

    // Reference line at 0
    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(ax(0), m.t); ctx.lineTo(ax(0), h - m.b); ctx.stroke();
    ctx.setLineDash([]);

    const slot = (h - m.t - m.b) / reps;
    for (let i = 0; i < reps; i++) {
      const y = m.t + slot * (i + 0.5);
      const [lo, hi] = intervals[i];
      const covers = lo <= 0 && hi >= 0;
      ctx.strokeStyle = covers ? ACCENT : BAD;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ax(lo), y); ctx.lineTo(ax(hi), y); ctx.stroke();
    }
  }
  for (const el of [nE, lE]) el.addEventListener('input', draw);
  go.addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();

// ============================================================
// 5) Hypothesis test
// ============================================================
(function demo_ht() {
  const cv = document.getElementById('cv-ht');
  const d  = document.getElementById('ht-d');
  const c  = document.getElementById('ht-c');
  const dv = document.getElementById('ht-dv'), cv2 = document.getElementById('ht-cv');
  const a  = document.getElementById('ht-a');
  const b  = document.getElementById('ht-b');
  const pw = document.getElementById('ht-p');

  function draw() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 28, r: 12, t: 14, b: 24 };
    const delta = +d.value, thr = +c.value;
    dv.textContent = delta.toFixed(2); cv2.textContent = thr.toFixed(2);

    const lo = Math.min(-4, -4 + delta - 2);
    const hi = Math.max(4, delta + 4);
    const ax = x => m.l + (x - lo) / (hi - lo) * (w - m.l - m.r);
    const ymax = 0.45;
    const ay = y => h - m.b - (y / ymax) * (h - m.t - m.b);

    const alpha = 1 - normCDF(thr);
    const beta  = normCDF(thr - delta);
    a.textContent = alpha.toFixed(3);
    b.textContent = beta.toFixed(3);
    pw.textContent = (1 - beta).toFixed(3);

    axes(ctx, w, h, m, { xLabel: 'test statistic' });

    function curve(mu, fill) {
      const pts = [];
      for (let i = 0; i <= 200; i++) {
        const x = lo + (hi - lo) * i / 200;
        pts.push([x, normPDF(x, mu, 1)]);
      }
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(ax(pts[0][0]), h - m.b);
      for (const [x, y] of pts) ctx.lineTo(ax(x), ay(y));
      ctx.lineTo(ax(pts[pts.length - 1][0]), h - m.b);
      ctx.fill();
      ctx.strokeStyle = fill === 'rgba(220,38,38,0.18)' ? BAD : ACCENT;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach(([x, y], i) => i ? ctx.lineTo(ax(x), ay(y)) : ctx.moveTo(ax(x), ay(y)));
      ctx.stroke();
    }

    // Draw H0 and H1 (very pale fill)
    curve(0, 'rgba(220,38,38,0.05)');
    curve(delta, 'rgba(67,56,202,0.05)');

    // Shade alpha (under H0, x>thr) and beta (under H1, x<thr)
    function shade(mu, from, to, color) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(ax(from), h - m.b);
      for (let i = 0; i <= 80; i++) {
        const x = from + (to - from) * i / 80;
        ctx.lineTo(ax(x), ay(normPDF(x, mu, 1)));
      }
      ctx.lineTo(ax(to), h - m.b); ctx.closePath();
      ctx.fill();
    }
    shade(0, thr, hi, 'rgba(220,38,38,0.45)');
    shade(delta, lo, thr, 'rgba(245,158,11,0.55)');

    // Threshold line
    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(ax(thr), m.t); ctx.lineTo(ax(thr), h - m.b); ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = INK; ctx.textAlign = 'center';
    ctx.fillText('H₀', ax(0), m.t + 10);
    ctx.fillText('H₁', ax(delta), m.t + 10);
  }
  for (const el of [d, c]) el.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();

// ============================================================
// 6) Linear regression (draggable points)
// ============================================================
(function demo_reg() {
  const cv = document.getElementById('cv-reg');
  const b1 = document.getElementById('reg-b1');
  const b0 = document.getElementById('reg-b0');
  const r2 = document.getElementById('reg-r2');
  const res = document.getElementById('reg-res');
  const rst = document.getElementById('reg-reset');

  let pts = [];
  function resetPts() {
    pts = [];
    for (let i = 0; i < 15; i++) {
      const x = -3 + 6 * lcg();
      const y = 0.6 * x + 0.5 + 0.7 * gauss();
      pts.push({ x, y });
    }
  }
  resetPts();

  const m = { l: 32, r: 12, t: 14, b: 24 };
  let W = 0, H = 0;
  let xmin = -4, xmax = 4, ymin = -4, ymax = 4;
  function ax(x) { return m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r); }
  function ay(y) { return H - m.b - (y - ymin) / (ymax - ymin) * (H - m.t - m.b); }
  function unax(px) { return xmin + (px - m.l) / (W - m.l - m.r) * (xmax - xmin); }
  function unay(py) { return ymin + (H - m.b - py) / (H - m.t - m.b) * (ymax - ymin); }

  let drag = null;

  function fit() {
    const n = pts.length;
    const mx = pts.reduce((s, p) => s + p.x, 0) / n;
    const my = pts.reduce((s, p) => s + p.y, 0) / n;
    let sxy = 0, sxx = 0, sst = 0;
    for (const p of pts) { sxy += (p.x - mx) * (p.y - my); sxx += (p.x - mx) ** 2; sst += (p.y - my) ** 2; }
    const slope = sxx ? sxy / sxx : 0;
    const intc  = my - slope * mx;
    let ssr = 0;
    for (const p of pts) { const e = p.y - (slope * p.x + intc); ssr += e * e; }
    return { slope, intc, r2: sst ? 1 - ssr / sst : 0 };
  }

  function draw() {
    const ctx = cv.getContext('2d');
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = cv.getBoundingClientRect();
    cv.width = Math.floor(rect.width * dpr);
    cv.height = Math.floor(parseInt(cv.getAttribute('height'), 10) * dpr);
    cv.style.height = parseInt(cv.getAttribute('height'), 10) + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    W = rect.width; H = parseInt(cv.getAttribute('height'), 10);

    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px Inter, sans-serif';
    axes(ctx, W, H, m, { xLabel: 'x', yLabel: 'y' });

    // grid
    ctx.strokeStyle = '#F2F2F5';
    for (let v = Math.ceil(xmin); v <= xmax; v++) {
      ctx.beginPath(); ctx.moveTo(ax(v), m.t); ctx.lineTo(ax(v), H - m.b); ctx.stroke();
    }
    for (let v = Math.ceil(ymin); v <= ymax; v++) {
      ctx.beginPath(); ctx.moveTo(m.l, ay(v)); ctx.lineTo(W - m.r, ay(v)); ctx.stroke();
    }

    const F = fit();
    b1.textContent = F.slope.toFixed(3);
    b0.textContent = F.intc.toFixed(3);
    r2.textContent = F.r2.toFixed(3);

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
        const yhat = F.slope * p.x + F.intc;
        ctx.beginPath(); ctx.moveTo(ax(p.x), ay(p.y)); ctx.lineTo(ax(p.x), ay(yhat)); ctx.stroke();
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
      if (Math.hypot(ax(p.x) - mx, ay(p.y) - my) <= 8) return i;
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
    pts[drag].x = unax(e.clientX - r.left);
    pts[drag].y = unay(e.clientY - r.top);
    draw();
  });
  cv.addEventListener('pointerup', () => { drag = null; });
  cv.addEventListener('pointerleave', () => { drag = null; });
  res.addEventListener('change', draw);
  rst.addEventListener('click', () => { resetPts(); draw(); });
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();

// ============================================================
// 7) Monte Carlo π
// ============================================================
(function demo_mc() {
  const cv = document.getElementById('cv-mc');
  const r  = document.getElementById('mc-r'), rv = document.getElementById('mc-rv');
  const ne = document.getElementById('mc-n');
  const ke = document.getElementById('mc-k');
  const pe = document.getElementById('mc-pi');
  const tg = document.getElementById('mc-toggle');
  const rs = document.getElementById('mc-reset');

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  let N = 0, K = 0;
  let running = true;
  let timer = null;

  function fit() {
    const rect = cv.getBoundingClientRect();
    cv.width = Math.floor(rect.width * dpr);
    cv.height = Math.floor(rect.width * dpr);
    cv.style.height = rect.width + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, S: rect.width };
  }

  let S = 0, ctx = null;
  function initFrame() {
    const f = fit(); ctx = f.ctx; S = f.S;
    ctx.fillStyle = '#FAFAFA';
    ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = INK;
    ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
    ctx.beginPath();
    ctx.moveTo(0, S); ctx.arc(0, S, S, -Math.PI / 2, 0); ctx.lineTo(0, S);
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2;
    ctx.stroke();
  }
  initFrame();

  function throwSome() {
    if (!ctx) return;
    const rate = +r.value;
    const batch = Math.max(1, Math.round(rate / 30));
    rv.textContent = `${rate}/s`;
    for (let i = 0; i < batch; i++) {
      const x = lcg(), y = lcg();
      const inside = x * x + y * y <= 1;
      ctx.fillStyle = inside ? ACCENT : MUTED;
      ctx.fillRect(x * S, (1 - y) * S, 1.5, 1.5);
      N++; if (inside) K++;
    }
    ne.textContent = N;
    ke.textContent = K;
    pe.textContent = N ? (4 * K / N).toFixed(5) : '—';
  }

  function loop() { if (running) throwSome(); }
  timer = setInterval(loop, 33);

  tg.addEventListener('click', () => {
    running = !running;
    tg.textContent = running ? 'pause' : 'resume';
  });
  rs.addEventListener('click', () => { N = 0; K = 0; ne.textContent = 0; ke.textContent = 0; pe.textContent = '—'; initFrame(); });
  r.addEventListener('input', () => { rv.textContent = `${r.value}/s`; });
  window.addEventListener('resize', () => { initFrame(); N = 0; K = 0; ne.textContent = 0; ke.textContent = 0; pe.textContent = '—'; });
})();

// ============================================================
// 8) Random walk
// ============================================================
(function demo_rw() {
  const cv = document.getElementById('cv-rw');
  const k  = document.getElementById('rw-k'), kv = document.getElementById('rw-kv');
  const b  = document.getElementById('rw-b'), bv = document.getElementById('rw-bv');
  const go = document.getElementById('rw-go');

  function draw() {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 28, r: 12, t: 14, b: 24 };
    const K = +k.value, B = +b.value;
    kv.textContent = K; bv.textContent = B.toFixed(2);
    const T = 200;
    const lim = Math.max(8, Math.sqrt(T) * 2 + Math.abs(B * T));
    const ax = t => m.l + t / T * (w - m.l - m.r);
    const ay = y => m.t + (1 - (y + lim) / (2 * lim)) * (h - m.t - m.b);
    axes(ctx, w, h, m, { xLabel: 't' });

    // Envelope ±√t
    ctx.strokeStyle = INK; ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let t = 0; t <= T; t++) {
      const y = ay(Math.sqrt(t) + B * t);
      t === 0 ? ctx.moveTo(ax(t), y) : ctx.lineTo(ax(t), y);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let t = 0; t <= T; t++) {
      const y = ay(-Math.sqrt(t) + B * t);
      t === 0 ? ctx.moveTo(ax(t), y) : ctx.lineTo(ax(t), y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // 0 line
    ctx.strokeStyle = RULE; ctx.beginPath();
    ctx.moveTo(m.l, ay(0)); ctx.lineTo(w - m.r, ay(0)); ctx.stroke();

    ctx.lineWidth = 1.2;
    for (let i = 0; i < K; i++) {
      ctx.strokeStyle = `hsla(${(i * 60) % 360}, 70%, 45%, 0.45)`;
      ctx.beginPath();
      let pos = 0;
      ctx.moveTo(ax(0), ay(0));
      for (let t = 1; t <= T; t++) {
        const step = (lcg() < 0.5 + B ? 1 : -1);
        pos += step;
        ctx.lineTo(ax(t), ay(pos));
      }
      ctx.stroke();
    }
  }
  for (const el of [k, b]) el.addEventListener('input', draw);
  go.addEventListener('click', draw);
  window.addEventListener('resize', draw);
  setTimeout(draw, 0);
})();
