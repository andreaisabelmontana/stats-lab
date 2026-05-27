# stats-lab

Visual probability & statistics. 13 self-contained demos.

**Live:** https://andreaisabelmontana.github.io/stats-lab/

1. Probability — Venn diagram with union, intersection, conditional, independence test
2. Discrete distributions (Bernoulli, Binomial, Geometric, Negative Binomial, Poisson) with E[X], Var[X], mode
3. Continuous distributions (Uniform, Normal, Exponential, Gamma, Beta, Chi-squared) + optional CDF
4. Sum of independent variables — Monte Carlo convolution
5. Central Limit Theorem
6. Bayes — 1000-person grid + base-rate fallacy
7. Maximum likelihood — drag the parameter, watch log L peak at the MLE
8. Confidence intervals — 100 CIs with coverage counter
9. Hypothesis test — α / β / power / p-value all visible
10. t-distribution vs Normal — heavy tails at low df
11. Linear regression — draggable points + 95% confidence band
12. Monte Carlo π
13. Random walks under √t envelope

Defensive helpers (`safe`, `clamp`, `robustMax`) keep every demo from crashing on degenerate inputs (Cauchy, Beta(0.5, 0.5), σ → 0, etc.).

Plain HTML + canvas + KaTeX. Indigo accent.

Part of the *-lab series: [calc-lab](https://github.com/andreaisabelmontana/calc-lab) · [algos-lab](https://github.com/andreaisabelmontana/algos-lab) · [sql-lab](https://github.com/andreaisabelmontana/sql-lab)
