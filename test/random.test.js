import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, gauss, expRV, unitSamplers, sampleMany } from '../src/stats/random.js';
import { mean, std } from '../src/stats/descriptive.js';

test('mulberry32 is deterministic per seed', () => {
  const a = mulberry32(99);
  const b = mulberry32(99);
  for (let i = 0; i < 10; i++) assert.equal(a(), b());
});

test('mulberry32 stays in [0, 1)', () => {
  const r = mulberry32(5);
  for (let i = 0; i < 10000; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1);
  }
});

test('LLN: sample mean of many seeded gauss draws approaches the true mean', () => {
  const rng = mulberry32(2024);
  const draws = sampleMany((r) => gauss(2, 3, r), 200000, rng);
  assert.ok(Math.abs(mean(draws) - 2) < 0.05, `mean ${mean(draws)}`);
  assert.ok(Math.abs(std(draws) - 3) < 0.05, `std ${std(draws)}`);
});

test('LLN: uniform sampler mean approaches 0.5', () => {
  const rng = mulberry32(11);
  const draws = sampleMany((r) => unitSamplers.uniform(r), 200000, rng);
  assert.ok(Math.abs(mean(draws) - 0.5) < 0.01, `mean ${mean(draws)}`);
});

test('LLN: exponential(λ) sample mean approaches 1/λ', () => {
  const rng = mulberry32(77);
  const lambda = 2;
  const draws = sampleMany((r) => expRV(lambda, r), 200000, rng);
  assert.ok(Math.abs(mean(draws) - 1 / lambda) < 0.02, `mean ${mean(draws)}`);
});
