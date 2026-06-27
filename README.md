# stats-lab

Visual probability & statistics — 13 self-contained, interactive canvas demos, backed by tested statistics modules.

**Live:** https://andreaisabelmontana.github.io/stats-lab/

## Demos

Probability (Venn) · discrete & continuous distributions · sums/convolution · Central Limit Theorem · Bayes (base-rate fallacy) · maximum likelihood · confidence intervals · hypothesis test (α/β/power/p-value) · t vs Normal · linear regression · Monte Carlo π · random walks.

Plain HTML + canvas + KaTeX. `index.html` reads sliders and renders; all the math lives in framework-free ES modules under `src/stats/`.

## Statistics cores

The numeric and statistical code is extracted into DOM-free ES modules, imported by the page and covered by unit tests.

| Module | Functions |
| --- | --- |
| `src/stats/numeric.js` | `logGamma`, `logBin`, `erf`, `safe`, `clamp` |
| `src/stats/descriptive.js` | `mean`, `variance`, `std`, `median`, `quantile`, `covariance`, `correlation` |
| `src/stats/distributions.js` | `normalPdf`, `normalCdf`, `zstar`, `bernoulliPmf`, `binomialPmf`, `geometricPmf`, `negBinomialPmf`, `poissonPmf`, `continuousPdf`, `continuousMoments`, `discreteMoments`, `tPdf`, `tCdf` |
| `src/stats/random.js` | `mulberry32` (seeded RNG), `gauss`, `expRV`, `bernoulliRV`, `unitSamplers`, `sampleMany` |
| `src/stats/regression.js` | `leastSquares`, `predict` |
| `src/stats/inference.js` | `confidenceInterval`, `covers`, `zTest`, `bayesDiagnostic` |

## Properties proven by the tests

- `mean`, `variance`, `std`, `median`, `quantile` match hand-computed values.
- Normal pdf integrates to ~1 and is symmetric; `normalCdf(0)=0.5`, `normalCdf(1.96)≈0.975`; `zstar(0.95)≈1.96`.
- Binomial, Poisson, geometric, negative-binomial and Bernoulli pmfs sum to ~1; continuous pdfs (uniform, normal, exponential, gamma, beta, χ²) integrate to ~1.
- Least squares recovers a known slope/intercept on linear data (exact, and under seeded noise); correlation is ±1 for perfect (anti-)linear relations.
- Sample mean of many seeded draws approaches the true mean (law of large numbers); seeded CIs cover the truth ~95% of the time.
- z-test α/β/power/p-value match the normal model; Bayes reproduces the base-rate fallacy.

## Run

Open `index.html` (or the live link) — no build step.

## Test

No dependencies. Requires Node 18+ (uses the built-in test runner).

```
node --test
```

```
ℹ tests 35
ℹ suites 0
ℹ pass 35
ℹ fail 0
ℹ duration_ms 274.6509
```

RNG is seeded (`mulberry32`) so the statistical tests are deterministic.

Part of the *-lab series: [calc-lab](https://github.com/andreaisabelmontana/calc-lab) · [algos-lab](https://github.com/andreaisabelmontana/algos-lab) · [sql-lab](https://github.com/andreaisabelmontana/sql-lab)
