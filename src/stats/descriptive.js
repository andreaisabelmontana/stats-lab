// descriptive.js — summary statistics over numeric arrays. No DOM.

/** Arithmetic mean. Empty array -> 0. */
export function mean(xs) {
  if (!xs.length) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/**
 * Variance. `sample = true` (default) divides by n-1 (Bessel-corrected,
 * unbiased estimator); `sample = false` divides by n (population variance).
 */
export function variance(xs, sample = true) {
  const n = xs.length;
  if (n < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return s / (sample ? n - 1 : n);
}

/** Standard deviation = sqrt(variance). */
export function std(xs, sample = true) {
  return Math.sqrt(variance(xs, sample));
}

/**
 * Quantile q in [0, 1] via linear interpolation between order statistics
 * (the "type 7" / R default rule). Operates on a sorted copy.
 */
export function quantile(xs, q) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  if (s.length === 1) return s[0];
  const pos = (s.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  return s[lo] + (pos - lo) * (s[hi] - s[lo]);
}

/** Median = 0.5 quantile. */
export function median(xs) {
  return quantile(xs, 0.5);
}

/**
 * Covariance between two equal-length arrays. `sample = true` divides by
 * n-1, otherwise by n.
 */
export function covariance(xs, ys, sample = true) {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  const mx = mean(xs.slice(0, n));
  const my = mean(ys.slice(0, n));
  let s = 0;
  for (let i = 0; i < n; i++) s += (xs[i] - mx) * (ys[i] - my);
  return s / (sample ? n - 1 : n);
}

/**
 * Pearson correlation coefficient in [-1, 1]. Returns 0 if either input
 * has zero variance.
 */
export function correlation(xs, ys) {
  const sx = std(xs);
  const sy = std(ys);
  if (sx === 0 || sy === 0) return 0;
  return covariance(xs, ys) / (sx * sy);
}
