import { test } from 'node:test';
import assert from 'node:assert/strict';
import { leastSquares, predict } from '../src/stats/regression.js';
import { mulberry32, gauss } from '../src/stats/random.js';

test('least squares recovers an exact line', () => {
  // y = 2x + 1, no noise
  const pts = [];
  for (let x = -5; x <= 5; x++) pts.push({ x, y: 2 * x + 1 });
  const f = leastSquares(pts);
  assert.ok(Math.abs(f.slope - 2) < 1e-12);
  assert.ok(Math.abs(f.intercept - 1) < 1e-12);
  assert.ok(Math.abs(f.r2 - 1) < 1e-12);
  assert.ok(f.rmse < 1e-12);
});

test('least squares recovers a known slope/intercept under seeded noise', () => {
  const rng = mulberry32(7);
  const trueSlope = -0.8, trueIntercept = 2.3;
  const pts = [];
  for (let i = 0; i < 4000; i++) {
    const x = -3 + 6 * rng();
    const y = trueSlope * x + trueIntercept + 0.3 * gauss(0, 1, rng);
    pts.push({ x, y });
  }
  const f = leastSquares(pts);
  assert.ok(Math.abs(f.slope - trueSlope) < 0.05, `slope ${f.slope}`);
  assert.ok(Math.abs(f.intercept - trueIntercept) < 0.05, `intercept ${f.intercept}`);
});

test('predict evaluates the fitted line', () => {
  const f = leastSquares([{ x: 0, y: 1 }, { x: 1, y: 3 }, { x: 2, y: 5 }]);
  assert.ok(Math.abs(predict(f, 3) - 7) < 1e-12);
});

test('degenerate input is handled', () => {
  const f = leastSquares([{ x: 1, y: 1 }]);
  assert.equal(f.slope, 0);
  assert.equal(f.intercept, 0);
});
