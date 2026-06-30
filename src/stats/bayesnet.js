// bayesnet.js — the canonical three-node "wet grass" Bayesian network and
// exact inference by full enumeration. No DOM. This is the smallest network
// that shows the two ideas the course turns on: conditional independence read
// off the graph, and "explaining away" — observing one cause makes a competing
// cause less likely. (Russell & Norvig's sprinkler/rain example.)
//
// Graph:  Rain → Sprinkler,  Rain → WetGrass,  Sprinkler → WetGrass.
// Rain and Sprinkler are marginally dependent (through Rain), and become
// dependent given WetGrass even though Sprinkler ⟂ nothing blocks them.

export const CPT = {
  R: 0.2,                                  // P(Rain = 1)
  S: { 0: 0.4, 1: 0.01 },                  // P(Sprinkler = 1 | Rain)
  W: { 0: { 0: 0.0, 1: 0.8 },              // P(WetGrass = 1 | Sprinkler, Rain)
       1: { 0: 0.9, 1: 0.99 } },
};

// Joint probability P(R=r, S=s, W=w) factorised along the graph.
export function joint(r, s, w, cpt = CPT) {
  const pr = r ? cpt.R : 1 - cpt.R;
  const ps = s ? cpt.S[r] : 1 - cpt.S[r];
  const pw = w ? cpt.W[s][r] : 1 - cpt.W[s][r];
  return pr * ps * pw;
}

// Posterior marginals P(node = 1 | evidence) for every node, by enumerating
// the 8 joint states consistent with the evidence. `evidence` keys are
// R / S / W with value 0, 1, or null (unobserved). Returns { R, S, W, Z }
// where Z is the probability of the evidence; on impossible evidence Z = 0
// and the marginals are null.
export function infer(evidence = {}, cpt = CPT) {
  const ev = { R: null, S: null, W: null, ...evidence };
  let Z = 0;
  const num = { R: 0, S: 0, W: 0 };
  for (let r = 0; r < 2; r++) {
    if (ev.R !== null && ev.R !== r) continue;
    for (let s = 0; s < 2; s++) {
      if (ev.S !== null && ev.S !== s) continue;
      for (let w = 0; w < 2; w++) {
        if (ev.W !== null && ev.W !== w) continue;
        const p = joint(r, s, w, cpt);
        Z += p;
        if (r === 1) num.R += p;
        if (s === 1) num.S += p;
        if (w === 1) num.W += p;
      }
    }
  }
  if (Z <= 0) return { R: null, S: null, W: null, Z: 0 };
  return { R: num.R / Z, S: num.S / Z, W: num.W / Z, Z };
}
