// numeric.js — framework-free numeric helpers shared across the stats cores.
// No DOM, no globals. Pure functions only.

export const TAU = Math.PI * 2;

/** Return x if finite, otherwise the default d. */
export const safe = (x, d = 0) => (Number.isFinite(x) ? x : d);

/** Clamp x into [a, b]. */
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/**
 * Lanczos approximation of log Γ(z), valid for all real z (uses the
 * reflection formula for z < 0.5). Underpins every factorial / binomial
 * coefficient computed in log space to avoid overflow.
 */
export function logGamma(z) {
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(TAU) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/** log of the binomial coefficient C(n, k). */
export function logBin(n, k) {
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1);
}

/**
 * Error function via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
 * Used to build the standard-normal CDF.
 */
export function erf(x) {
  const s = Math.sign(x);
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return s * y;
}
