// regression.js — ordinary least-squares for simple linear regression.
// No DOM. Takes arrays of {x, y} points (as the draggable demo uses).

import { mean } from './descriptive.js';

/**
 * Fit y = slope·x + intercept by minimising Σ(y - ŷ)².
 * Returns slope, intercept, R², RMSE, plus Sxx / SSE / mean-x for the
 * confidence-band geometry the demo draws.
 */
export function leastSquares(points) {
  const n = points.length;
  if (n < 2) return { slope: 0, intercept: 0, r2: 0, rmse: 0, sxx: 0, sse: 0, meanX: 0 };

  const mx = mean(points.map((p) => p.x));
  const my = mean(points.map((p) => p.y));

  let sxy = 0, sxx = 0, sst = 0;
  for (const p of points) {
    sxy += (p.x - mx) * (p.y - my);
    sxx += (p.x - mx) ** 2;
    sst += (p.y - my) ** 2;
  }
  const slope = sxx ? sxy / sxx : 0;
  const intercept = my - slope * mx;

  let sse = 0;
  for (const p of points) {
    const e = p.y - (slope * p.x + intercept);
    sse += e * e;
  }

  return {
    slope,
    intercept,
    r2: sst ? 1 - sse / sst : 0,
    rmse: Math.sqrt(sse / n),
    sxx,
    sse,
    meanX: mx,
  };
}

/** Predict ŷ at x from a fitted model. */
export function predict(fit, x) {
  return fit.slope * x + fit.intercept;
}
