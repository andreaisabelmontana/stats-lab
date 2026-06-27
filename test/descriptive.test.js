import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mean, variance, std, median, quantile, covariance, correlation } from '../src/stats/descriptive.js';

test('mean matches hand-computed value', () => {
  assert.equal(mean([1, 2, 3, 4, 5]), 3);
  assert.equal(mean([2, 4, 6]), 4);
  assert.equal(mean([]), 0);
});

test('population variance matches hand-computed value', () => {
  // data {2,4,4,4,5,5,7,9}: mean 5, sum sq dev 32, /8 = 4
  assert.equal(variance([2, 4, 4, 4, 5, 5, 7, 9], false), 4);
});

test('sample variance uses n-1', () => {
  // data {2,4,6}: mean 4, sum sq dev 8, /(3-1) = 4
  assert.equal(variance([2, 4, 6], true), 4);
});

test('std is sqrt of population variance', () => {
  assert.equal(std([2, 4, 4, 4, 5, 5, 7, 9], false), 2);
});

test('median for odd and even length', () => {
  assert.equal(median([3, 1, 2]), 2);          // odd -> middle
  assert.equal(median([1, 2, 3, 4]), 2.5);     // even -> average of middles
  assert.equal(median([7]), 7);
});

test('quantile interpolates (type-7) like R default', () => {
  const xs = [0, 1, 2, 3, 4];
  assert.equal(quantile(xs, 0), 0);
  assert.equal(quantile(xs, 1), 4);
  assert.equal(quantile(xs, 0.5), 2);
  assert.equal(quantile(xs, 0.25), 1);
  assert.equal(quantile(xs, 0.75), 3);
});

test('covariance and correlation on perfectly linear data', () => {
  const x = [1, 2, 3, 4, 5];
  const y = x.map((v) => 2 * v + 1);           // exactly linear, slope > 0
  assert.ok(Math.abs(correlation(x, y) - 1) < 1e-12);
  // cov = slope * var(x); sample var(x) = 2.5 -> cov = 5
  assert.ok(Math.abs(covariance(x, y) - 5) < 1e-12);
});

test('correlation is -1 for perfect anti-linear data', () => {
  const x = [1, 2, 3, 4, 5];
  const y = x.map((v) => -3 * v + 10);
  assert.ok(Math.abs(correlation(x, y) + 1) < 1e-12);
});

test('correlation is 0 when a series has no variance', () => {
  assert.equal(correlation([1, 2, 3], [5, 5, 5]), 0);
});
