// markov.js — finite Markov chains: one transition step and the stationary
// (steady-state) distribution by power iteration. No DOM.
//
// State is a row vector π (probabilities summing to 1); a step is the
// vector–matrix product πₜ₊₁ = πₜ P, where P is row-stochastic (each row a
// distribution over next states). For an ergodic chain the distribution
// converges to the unique π* solving π* = π* P, regardless of where it starts.

// One transition: returns the next distribution πP.
export function step(pi, P) {
  const m = P.length;
  const out = new Array(m).fill(0);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) out[j] += pi[i] * P[i][j];
  return out;
}

// True if every row of P sums to 1 (row-stochastic).
export function isStochastic(P, tol = 1e-9) {
  return P.every(row => Math.abs(row.reduce((s, x) => s + x, 0) - 1) < tol);
}

// Stationary distribution by power iteration from the uniform start, stopping
// when the L1 change falls below `tol`. Returns { dist, iters, converged }.
export function stationary(P, { tol = 1e-12, maxIters = 5000 } = {}) {
  const m = P.length;
  let pi = new Array(m).fill(1 / m);
  let iters = 0, converged = false;
  for (; iters < maxIters; iters++) {
    const next = step(pi, P);
    let diff = 0;
    for (let j = 0; j < m; j++) diff += Math.abs(next[j] - pi[j]);
    pi = next;
    if (diff < tol) { converged = true; iters++; break; }
  }
  return { dist: pi, iters, converged };
}
