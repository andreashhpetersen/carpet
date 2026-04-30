# Related Work: Policy Extraction with Non-Axis-Aligned Boundaries

## Key finding (2026-04-29)

No published work uses **polynomial boundaries as split criteria** in decision trees
for RL policy extraction. The field has moved axis-aligned → linear oblique, but
not further. This appears to be a genuine novelty of CARPET.

---

## Papers to cite / position against

### Direct baseline
- **VIPER** — Bastani, Pu, Solar-Lezama. *Verifiable Reinforcement Learning via
  Policy Extraction.* NeurIPS 2018. https://arxiv.org/abs/1805.08328
  - DAgger-style imitation into axis-aligned CART trees.
  - Action mapping only — no transition structure.

### Linear oblique splits (closest boundary expressiveness)
- **Iterative Oblique Decision Trees** — *Algorithms* (MDPI), 2023.
  https://www.mdpi.com/1999-4893/16/6/282
  - Iterative training outperforming VIPER-style sampling. Linear oblique only.

- **Enhanced Oblique DT for DRL** (power systems) — *Electric Power Systems
  Research*, 2022. https://www.sciencedirect.com/science/article/abs/pii/S0378779622001626
  - IGR-WODT: information-gain-rate weighted oblique splits. Linear only.

### Differentiable / soft trees (most natural place to extend to polynomial)
- **Differentiable Decision Trees** — Silva et al. AISTATS 2020.
  https://proceedings.mlr.press/v108/silva20a.html
  - Soft probabilistic routing; gradient-based optimisation. Linear sigmoids —
    polynomial heads would be the natural next step but not taken.

- **Distill2Explain** — E-Energy 2024. https://arxiv.org/html/2403.11907v1
  - Differentiable DT distillation for energy controllers. Linear splits.

### Other notable work
- **Neural-to-Tree Policy Distillation** — https://arxiv.org/pdf/2108.06898
  - Policy improvement criterion for distillation. Axis-aligned.

- **INTERPRETER** — Kohler, Delfosse et al., 2024.
  https://arxiv.org/html/2405.14956v1
  - Interpretable/editable programmatic tree policies. Linear oblique.

- **MoET** (Mixture of Expert Trees) — OpenReview.
  https://openreview.net/forum?id=BJlxdCVKDB
  - Soft gating + multiple tree experts. Individual splits remain linear.

- **DTPO** — https://arxiv.org/abs/2408.11632
  - Direct policy-gradient optimisation of a decision tree. Axis-aligned.

- **MENS-DT-RL** — *Artificial Intelligence* (Elsevier), 2023.
  https://www.sciencedirect.com/science/article/abs/pii/S0004370223002035
  - Evolutionary approach to grow axis-aligned trees.

---

## Positioning argument for the paper

VIPER-style work captures the action mapping only, and even the oblique extensions
stop at linear boundaries. Polynomial splits are strictly more expressive while
remaining interpretable in terms of state variables — unlike neural/soft trees.
CARPET also differs fundamentally in that it targets **transition structure**, not
just the action mapping.

---

## TODO before submission
- Manual arXiv search for concurrent unpublished work (polynomial + policy tree).
