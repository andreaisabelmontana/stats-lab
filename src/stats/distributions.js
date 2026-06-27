// distributions.js — pdf/pmf/cdf and moments for the families the lab draws.
// Pure math, no DOM. Mirrors the formulas used by the canvas demos.

import { TAU, safe, clamp, logGamma, logBin, erf } from './numeric.js';

// ---------- standard normal ----------------------------------------------

/** Normal pdf at x with mean mu, std sd. */
export function normalPdf(x, mu = 0, sd = 1) {
  const z = safe((x - mu) / sd, 0);
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(TAU));
}

/** Standard-normal CDF Φ(z). */
export function normalCdf(z) {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

/**
 * Inverse standard-normal "z*" for a two-sided confidence level
 * (e.g. 0.95 -> ~1.96). Newton iteration on Φ(z) = (1+level)/2.
 */
export function zstar(level) {
  const target = (1 + clamp(level, 0.5, 0.999)) / 2;
  let z = 1.96;
  for (let i = 0; i < 40; i++) {
    const f = normalCdf(z) - target;
    const df = normalPdf(z, 0, 1);
    if (!df) break;
    z -= f / df;
  }
  return z;
}

// ---------- discrete pmfs ------------------------------------------------

/** Bernoulli(p) pmf at k ∈ {0, 1}. */
export function bernoulliPmf(k, p) {
  p = clamp(p, 1e-6, 1 - 1e-6);
  if (k === 1) return p;
  if (k === 0) return 1 - p;
  return 0;
}

/** Binomial(n, p) pmf at k. */
export function binomialPmf(k, n, p) {
  if (k < 0 || k > n) return 0;
  p = clamp(p, 1e-6, 1 - 1e-6);
  const lp = logBin(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p);
  return safe(Math.exp(lp), 0);
}

/** Geometric(p) pmf — number of trials until first success, k = 1, 2, … */
export function geometricPmf(k, p) {
  if (k < 1) return 0;
  p = clamp(p, 1e-6, 1 - 1e-6);
  return p * Math.pow(1 - p, k - 1);
}

/** Negative-binomial pmf: k failures before the r-th success. */
export function negBinomialPmf(k, r, p) {
  if (k < 0) return 0;
  p = clamp(p, 1e-6, 1 - 1e-6);
  const lp = logGamma(k + r) - logGamma(k + 1) - logGamma(r)
           + r * Math.log(p) + k * Math.log(1 - p);
  return safe(Math.exp(lp), 0);
}

/** Poisson(λ) pmf at k. */
export function poissonPmf(k, lambda) {
  if (k < 0) return 0;
  lambda = Math.max(1e-9, lambda);
  const lp = -lambda + k * Math.log(lambda) - logGamma(k + 1);
  return safe(Math.exp(lp), 0);
}

/** Mean and variance of each discrete family. */
export function discreteMoments(fam, p1, p2) {
  if (fam === 'bernoulli') { const p = clamp(p2, 1e-6, 1 - 1e-6); return { mu: p, vr: p * (1 - p) }; }
  if (fam === 'binomial')  { const n = Math.max(1, Math.round(p1)), p = clamp(p2, 1e-6, 1 - 1e-6); return { mu: n * p, vr: n * p * (1 - p) }; }
  if (fam === 'geometric') { const p = clamp(p2, 1e-6, 1 - 1e-6); return { mu: 1 / p, vr: (1 - p) / (p * p) }; }
  if (fam === 'negbin')    { const r = Math.max(1, Math.round(p1)), p = clamp(p2, 1e-6, 1 - 1e-6); return { mu: r * (1 - p) / p, vr: r * (1 - p) / (p * p) }; }
  if (fam === 'poisson')   { const lam = Math.max(1e-9, p1); return { mu: lam, vr: lam }; }
  return { mu: 0, vr: 0 };
}

// ---------- continuous pdfs ----------------------------------------------

/**
 * pdf of a continuous family. `a`, `b` are the family parameters
 * (see CFG in app.js): uniform(a,b), normal(μ,σ), exponential(λ),
 * gamma(k,θ), beta(α,β), chi2(k).
 */
export function continuousPdf(fam, x, a, b) {
  if (fam === 'uniform') {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    return (x >= lo && x <= hi && hi > lo) ? 1 / (hi - lo) : 0;
  }
  if (fam === 'normal') return normalPdf(x, a, b);
  if (fam === 'exponential') return x < 0 ? 0 : a * Math.exp(-a * x);
  if (fam === 'gamma') {
    if (x <= 0) return 0;
    const lp = (a - 1) * Math.log(x) - x / b - a * Math.log(b) - logGamma(a);
    return safe(Math.exp(lp), 0);
  }
  if (fam === 'beta') {
    if (x <= 0 || x >= 1) return 0;
    const lp = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)
             + logGamma(a + b) - logGamma(a) - logGamma(b);
    return safe(Math.exp(lp), 0);
  }
  if (fam === 'chi2') {
    if (x <= 0) return 0;
    const k2 = a / 2;
    const lp = (k2 - 1) * Math.log(x) - x / 2 - k2 * Math.log(2) - logGamma(k2);
    return safe(Math.exp(lp), 0);
  }
  return 0;
}

/** Mean and variance of each continuous family. */
export function continuousMoments(fam, a, b) {
  if (fam === 'uniform')     { const lo = Math.min(a, b), hi = Math.max(a, b); return { mu: (lo + hi) / 2, vr: (hi - lo) ** 2 / 12 }; }
  if (fam === 'normal')      return { mu: a, vr: b * b };
  if (fam === 'exponential') return { mu: 1 / a, vr: 1 / (a * a) };
  if (fam === 'gamma')       return { mu: a * b, vr: a * b * b };
  if (fam === 'beta')        return { mu: a / (a + b), vr: a * b / ((a + b) ** 2 * (a + b + 1)) };
  if (fam === 'chi2')        return { mu: a, vr: 2 * a };
  return { mu: 0, vr: 0 };
}

// ---------- Student's t --------------------------------------------------

/** Student-t pdf at x with df degrees of freedom. */
export function tPdf(x, df) {
  const c = logGamma((df + 1) / 2) - logGamma(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(c) * Math.pow(1 + x * x / df, -(df + 1) / 2);
}

/**
 * Student-t CDF via composite Simpson integration of the pdf from -8 to c.
 * Accurate enough for the tail-mass readouts in the lab.
 */
export function tCdf(c, df) {
  let s = 0;
  const N = 400;
  const lo = -8, hi = c, dx = (hi - lo) / N;
  for (let i = 0; i <= N; i++) {
    const x = lo + i * dx;
    const w = (i === 0 || i === N) ? 1 : (i % 2 ? 4 : 2);
    s += w * tPdf(x, df);
  }
  return Math.min(1, Math.max(0, s * dx / 3));
}
