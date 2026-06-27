// random.js — seedable RNG and the samplers the simulations draw from.
// Every sampler takes an explicit `rng` (a () => number in [0,1)) so the
// demos stay reproducible under test. Default is Math.random.

import { TAU } from './numeric.js';

/**
 * mulberry32 — a tiny, fast, well-distributed 32-bit PRNG.
 * Returns a function producing floats in [0, 1). Deterministic per seed.
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal (Box–Muller), scaled to N(mu, sd). */
export function gauss(mu = 0, sd = 1, rng = Math.random) {
  const u = Math.max(1e-12, rng()), v = rng();
  return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
}

/** Exponential(λ) draw via inverse-CDF. */
export function expRV(lambda, rng = Math.random) {
  return -Math.log(Math.max(1e-12, rng())) / lambda;
}

/** Bernoulli(p) draw -> 0 or 1. */
export function bernoulliRV(p, rng = Math.random) {
  return rng() < p ? 1 : 0;
}

/** Named unit samplers used by the "sum of variables" demo. */
export const unitSamplers = {
  uniform: (rng = Math.random) => rng(),
  normal: (rng = Math.random) => gauss(0, 1, rng),
  exp: (rng = Math.random) => expRV(1, rng),
  tri: (rng = Math.random) => rng() + rng() - 1,
};

/**
 * Draw `n` samples from a sampler `fn(rng)` using a seeded generator.
 * Convenience for tests and the LLN demo.
 */
export function sampleMany(fn, n, rng = Math.random) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(rng);
  return out;
}
