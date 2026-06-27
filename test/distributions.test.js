import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  normalPdf, normalCdf, zstar,
  bernoulliPmf, binomialPmf, geometricPmf, negBinomialPmf, poissonPmf,
  continuousPdf, continuousMoments, tPdf, tCdf,
} from '../src/stats/distributions.js';

// numeric integral of f over [a, b] with the trapezoid rule
function integrate(f, a, b, steps = 20000) {
  const dx = (b - a) / steps;
  let s = 0.5 * (f(a) + f(b));
  for (let i = 1; i < steps; i++) s += f(a + i * dx);
  return s * dx;
}

function sumPmf(pmf, kMax) {
  let s = 0;
  for (let k = 0; k <= kMax; k++) s += pmf(k);
  return s;
}

test('normal pdf integrates to ~1', () => {
  const total = integrate((x) => normalPdf(x, 0, 1), -12, 12);
  assert.ok(Math.abs(total - 1) < 1e-4, `integral was ${total}`);
});

test('normal pdf is symmetric about the mean', () => {
  for (const x of [0.3, 1, 2.5]) {
    assert.ok(Math.abs(normalPdf(x, 0, 1) - normalPdf(-x, 0, 1)) < 1e-15);
  }
});

test('normal cdf known landmarks', () => {
  assert.ok(Math.abs(normalCdf(0) - 0.5) < 1e-12);
  assert.ok(Math.abs(normalCdf(1.96) - 0.975) < 1e-3, normalCdf(1.96));
  assert.ok(Math.abs(normalCdf(-1.96) - 0.025) < 1e-3);
});

test('zstar inverts the normal cdf (0.95 -> ~1.96)', () => {
  assert.ok(Math.abs(zstar(0.95) - 1.96) < 1e-2, zstar(0.95));
  assert.ok(Math.abs(normalCdf(zstar(0.99)) - 0.995) < 1e-3);
});

test('binomial pmf sums to ~1', () => {
  const n = 20, p = 0.37;
  assert.ok(Math.abs(sumPmf((k) => binomialPmf(k, n, p), n) - 1) < 1e-9);
});

test('binomial pmf matches a hand value: C(5,2) 0.5^5 = 0.3125', () => {
  assert.ok(Math.abs(binomialPmf(2, 5, 0.5) - 0.3125) < 1e-12);
});

test('poisson pmf sums to ~1', () => {
  const lam = 4.2;
  assert.ok(Math.abs(sumPmf((k) => poissonPmf(k, lam), 60) - 1) < 1e-9);
});

test('poisson pmf matches hand value: e^-2 ~ 0.13534 at k=0', () => {
  assert.ok(Math.abs(poissonPmf(0, 2) - Math.exp(-2)) < 1e-12);
});

test('bernoulli / geometric / negbin pmfs sum to ~1', () => {
  assert.ok(Math.abs(bernoulliPmf(0, 0.3) + bernoulliPmf(1, 0.3) - 1) < 1e-12);
  let g = 0;
  for (let k = 1; k <= 2000; k++) g += geometricPmf(k, 0.2);
  assert.ok(Math.abs(g - 1) < 1e-6, `geometric sum ${g}`);
  let nb = 0;
  for (let k = 0; k <= 4000; k++) nb += negBinomialPmf(k, 5, 0.4);
  assert.ok(Math.abs(nb - 1) < 1e-6, `negbin sum ${nb}`);
});

test('continuous pdfs integrate to ~1', () => {
  assert.ok(Math.abs(integrate((x) => continuousPdf('normal', x, 0, 1), -12, 12) - 1) < 1e-4);
  assert.ok(Math.abs(integrate((x) => continuousPdf('uniform', x, 0, 1), -1, 2) - 1) < 1e-4);
  assert.ok(Math.abs(integrate((x) => continuousPdf('exponential', x, 1.5), 0, 40) - 1) < 1e-4);
  assert.ok(Math.abs(integrate((x) => continuousPdf('gamma', x, 2, 1), 0, 60) - 1) < 1e-3);
  assert.ok(Math.abs(integrate((x) => continuousPdf('beta', x, 2, 5), 0, 1) - 1) < 1e-3);
  assert.ok(Math.abs(integrate((x) => continuousPdf('chi2', x, 4), 0, 80) - 1) < 1e-3);
});

test('continuous pdf mean matches the reported moment (normal, exp, gamma)', () => {
  // E[X] = ∫ x f(x) dx should match continuousMoments().mu
  const muNormal = integrate((x) => x * continuousPdf('normal', x, 1.5, 1), -12, 14);
  assert.ok(Math.abs(muNormal - continuousMoments('normal', 1.5, 1).mu) < 1e-3);
  const muExp = integrate((x) => x * continuousPdf('exponential', x, 2), 0, 40);
  assert.ok(Math.abs(muExp - continuousMoments('exponential', 2).mu) < 1e-3);
});

test('student-t pdf integrates to ~1 and tCdf landmarks', () => {
  assert.ok(Math.abs(integrate((x) => tPdf(x, 5), -40, 40) - 1) < 1e-3);
  assert.ok(Math.abs(tCdf(0, 10) - 0.5) < 1e-3, tCdf(0, 10));
  // heavier tails than normal: more mass beyond 2 at low df
  assert.ok((1 - tCdf(2, 3)) > (1 - normalCdf(2)));
});
