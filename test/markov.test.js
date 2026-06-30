import { test } from 'node:test';
import assert from 'node:assert/strict';
import { step, stationary, isStochastic } from '../src/stats/markov.js';

const sum = a => a.reduce((s, x) => s + x, 0);

test('a step keeps the distribution normalised', () => {
  const P = [[0.7, 0.3], [0.4, 0.6]];
  const next = step([1, 0], P);
  assert.ok(Math.abs(sum(next) - 1) < 1e-12, `sum ${sum(next)}`);
  assert.deepEqual(next, [0.7, 0.3]);   // starting in state 0 gives that row
});

test('isStochastic accepts valid rows and rejects bad ones', () => {
  assert.ok(isStochastic([[0.5, 0.5], [0.2, 0.8]]));
  assert.ok(!isStochastic([[0.5, 0.6], [0.2, 0.8]]));
});

test('stationary matches the closed form for a 2-state chain', () => {
  // P = [[1-a, a],[b, 1-b]] ⇒ π* = [b/(a+b), a/(a+b)].
  const a = 0.3, b = 0.2;
  const { dist, converged } = stationary([[1 - a, a], [b, 1 - b]]);
  assert.ok(converged, 'should converge');
  assert.ok(Math.abs(dist[0] - b / (a + b)) < 1e-9, `π0 ${dist[0]}`);
  assert.ok(Math.abs(dist[1] - a / (a + b)) < 1e-9, `π1 ${dist[1]}`);
});

test('the stationary distribution is a fixed point: π* P = π*', () => {
  const P = [[0.1, 0.6, 0.3], [0.4, 0.1, 0.5], [0.5, 0.4, 0.1]];
  const { dist } = stationary(P);
  const after = step(dist, P);
  for (let j = 0; j < 3; j++) assert.ok(Math.abs(after[j] - dist[j]) < 1e-9, `drift at ${j}`);
  assert.ok(Math.abs(sum(dist) - 1) < 1e-12);
});

test('a symmetric (doubly-stochastic) chain has the uniform stationary', () => {
  const P = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]];
  const { dist } = stationary(P);
  for (const p of dist) assert.ok(Math.abs(p - 1 / 3) < 1e-9, `not uniform: ${dist}`);
});
