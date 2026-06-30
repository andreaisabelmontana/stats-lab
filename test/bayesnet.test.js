import { test } from 'node:test';
import assert from 'node:assert/strict';
import { joint, infer, CPT } from '../src/stats/bayesnet.js';

test('the joint distribution sums to 1', () => {
  let s = 0;
  for (let r = 0; r < 2; r++) for (let sp = 0; sp < 2; sp++) for (let w = 0; w < 2; w++) s += joint(r, sp, w);
  assert.ok(Math.abs(s - 1) < 1e-12, `total ${s}`);
});

test('priors match the network (no evidence)', () => {
  const p = infer({});
  assert.ok(Math.abs(p.R - 0.2) < 1e-12, `P(R) ${p.R}`);
  // P(S=1) = P(S|R=1)P(R=1) + P(S|R=0)P(R=0) = .01*.2 + .4*.8 = .322
  assert.ok(Math.abs(p.S - 0.322) < 1e-12, `P(S) ${p.S}`);
  assert.ok(Math.abs(p.Z - 1) < 1e-12, 'Z is 1 with no evidence');
});

test('observed nodes collapse to their evidence', () => {
  assert.equal(infer({ R: 1 }).R, 1);
  assert.equal(infer({ S: 0 }).S, 0);
});

test('wet grass raises the probability of rain above its prior', () => {
  assert.ok(infer({ W: 1 }).R > 0.2, 'seeing wet grass should make rain more likely');
});

test('explaining away: a known sprinkler lowers the chance of rain', () => {
  const withWet = infer({ W: 1 }).R;
  const withWetAndSprinkler = infer({ W: 1, S: 1 }).R;
  assert.ok(withWetAndSprinkler < withWet,
    `explaining away failed: ${withWetAndSprinkler} should be < ${withWet}`);
});

test('marginal independence of Rain and Sprinkler is broken by their child', () => {
  // R and S are NOT independent given W (common effect) — observing S changes P(R|W).
  const base = infer({ W: 1 }).R;
  const givenS = infer({ W: 1, S: 1 }).R;
  assert.notEqual(base, givenS);
});
