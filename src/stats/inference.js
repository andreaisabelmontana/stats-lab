// inference.js — confidence intervals, a z-test, and Bayes posteriors.
// No DOM. Builds on the standard-normal helpers in distributions.js.

import { mean, std } from './descriptive.js';
import { normalCdf, zstar } from './distributions.js';

/**
 * Two-sided confidence interval for a mean from a sample, using the
 * normal critical value z*. Returns the point estimate, half-width and
 * the [lo, hi] bounds.
 */
export function confidenceInterval(sample, level = 0.95) {
  const n = sample.length;
  const m = mean(sample);
  const s = std(sample, true);
  const z = zstar(level);
  const half = (z * s) / Math.sqrt(n);
  return { mean: m, half, lo: m - half, hi: m + half, z };
}

/** Whether an interval {lo, hi} covers a value (default 0). */
export function covers(interval, value = 0) {
  return interval.lo <= value && interval.hi >= value;
}

/**
 * One-sided z-test geometry for H0: μ=0 vs H1: μ=δ, both with unit
 * variance, rejecting when z > threshold.
 *   alpha = P(reject | H0)            = 1 - Φ(threshold)
 *   beta  = P(fail to reject | H1)    = Φ(threshold - δ)
 *   power = 1 - beta
 *   pValue (for an observed z)        = 1 - Φ(z_obs)
 */
export function zTest(delta, threshold, zObs) {
  const alpha = 1 - normalCdf(threshold);
  const beta = normalCdf(threshold - delta);
  return { alpha, beta, power: 1 - beta, pValue: 1 - normalCdf(zObs) };
}

/**
 * Bayes for a diagnostic test on a population of `n` people.
 *   prior       = P(disease)
 *   sensitivity = P(+ | disease)        (true-positive rate)
 *   specificity = P(- | no disease)     (true-negative rate)
 * Returns the 2x2 counts and the posteriors P(D|+) and P(D|-).
 */
export function bayesDiagnostic(prior, sensitivity, specificity, n = 1000) {
  const diseased = Math.round(n * prior);
  const healthy = n - diseased;
  const tp = Math.round(diseased * sensitivity);
  const fn = diseased - tp;
  const fp = Math.round(healthy * (1 - specificity));
  const tn = healthy - fp;
  const posteriorPos = tp + fp > 0 ? tp / (tp + fp) : 0;     // P(D | +)
  const posteriorNeg = tn + fn > 0 ? fn / (tn + fn) : 0;     // P(D | -)
  return { tp, fn, fp, tn, posteriorPos, posteriorNeg };
}
