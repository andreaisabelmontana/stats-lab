import { test } from 'node:test';
import assert from 'node:assert/strict';
import { confidenceInterval, covers, zTest, bayesDiagnostic } from '../src/stats/inference.js';
import { mulberry32, gauss } from '../src/stats/random.js';

test('confidence interval is centred on the sample mean', () => {
  const ci = confidenceInterval([1, 2, 3, 4, 5], 0.95);
  assert.equal(ci.mean, 3);
  assert.ok(ci.lo < 3 && ci.hi > 3);
  assert.ok(Math.abs((ci.hi - ci.lo) / 2 - ci.half) < 1e-12);
});

test('95% intervals cover the true mean ~95% of the time (seeded)', () => {
  const rng = mulberry32(123);
  const reps = 4000, n = 30;
  let covered = 0;
  for (let i = 0; i < reps; i++) {
    const sample = Array.from({ length: n }, () => gauss(0, 1, rng));
    if (covers(confidenceInterval(sample, 0.95), 0)) covered++;
  }
  const rate = covered / reps;
  // normal-approx CI on n=30 is a touch narrow, so allow a band around 0.95
  assert.ok(rate > 0.9 && rate < 0.98, `coverage ${rate}`);
});

test('z-test alpha/beta/power/p-value match the normal model', () => {
  // threshold 1.645 -> alpha ~ 0.05
  const r = zTest(2, 1.645, 2.1);
  assert.ok(Math.abs(r.alpha - 0.05) < 1e-3, `alpha ${r.alpha}`);
  assert.ok(Math.abs(r.power - (1 - r.beta)) < 1e-12);
  assert.ok(r.power > 0 && r.power < 1);
  // p-value at z=2.1 is the upper tail 1 - Φ(2.1) ~ 0.0179
  assert.ok(Math.abs(r.pValue - 0.0179) < 2e-3, `p ${r.pValue}`);
});

test('larger effect size raises power', () => {
  const small = zTest(1, 1.645, 1.645).power;
  const large = zTest(3, 1.645, 1.645).power;
  assert.ok(large > small);
});

test('bayes base-rate fallacy: rare disease stays unlikely after a positive', () => {
  // prior 1%, sens 99%, spec 99% -> P(D|+) = 50%
  const b = bayesDiagnostic(0.01, 0.99, 0.99, 1000);
  assert.equal(b.tp, 10);
  assert.equal(b.fp, 10);
  assert.ok(Math.abs(b.posteriorPos - 0.5) < 1e-9);
  // counts partition the population
  assert.equal(b.tp + b.fn + b.fp + b.tn, 1000);
});
