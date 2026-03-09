# Kappa-FIN

**Topological Early Warning System for Financial Market Crises**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18883821.svg)](https://doi.org/10.5281/zenodo.18883821)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Part of Kappa-Method](https://img.shields.io/badge/Kappa--Method-ecosystem-brightgreen)](https://github.com/aprimora-ai/Kappa-Method)

Kappa-FIN applies the **Kappa Method** — persistent homology H₁ + Forman-Ricci curvature on rolling correlation networks — to detect systemic risk precursors in financial markets weeks to months before crises materialize. Version 3 extends validation to **17 historical scenarios (1985–2023)** and formalizes the **Structural Pressurization Signature (SPS)** as a prospective, falsifiable detection criterion over the full five-observable vector.

**Author:** David Ohio · [odavidohio@gmail.com](mailto:odavidohio@gmail.com) · Independent Researcher  
**Repository:** [github.com/aprimora-ai/Kappa-FIN](https://github.com/aprimora-ai/Kappa-FIN)  
**Paper (v3):** [`paper/artigo_kappa_fin_en_v3.pdf`](paper/artigo_kappa_fin_en_v3.pdf)

---

## Core Hypothesis

Financial crises do not emerge from chaos — they emerge from **organized topological coherence**.

As asset prices decouple from fundamentals, the correlation network transitions from a sparse, diverse, dissipative configuration to a rigid, memory-laden one. The Kappa-FIN method tracks this transition continuously through five structural observables:

| Observable | Symbol | Meaning |
|---|---|---|
| **Ohio Number** | `Oh(t)` | Topological pressure relative to the healthy baseline |
| **Structural Memory** | `Φ(t)` | Accumulated curvature negativity above CALM floor |
| **Dynamic Rigidity** | `η(t)` | Resistance to structural reorganization |
| **Structural Diversity** | `Ξ(t)` | Active topological degrees of freedom |
| **State-Phase Divergence** | `DEF(t)` | Self-consistency of the system's structural trajectory |

A **Structural Pressurization Signature (SPS)** — simultaneous activation of all five observables in the direction of structural lock-in — constitutes a confirmed regime transition warning, prospective and independent of the event's cause.

---

## Method Overview

```
Market prices (yfinance)
        │
        ▼
Log-returns  ──► Rolling 22-day window
        │
        ▼
Spearman correlation + Ledoit-Wolf shrinkage ──► Distance matrix
        │
        ▼
k-NN graph (k=5)
    ├── Forman-Ricci curvature  ──► η(t) = 1 / (|κ̄| + floor)
    └── Rips complex (GUDHI)
            │
            ▼
    H₁ persistent homology ──► entropy, dominance, β₁, total mass
            │
            ▼
CALM ensemble identification (KCPT penalty v4.6, median aggregation)
        │
        ▼
    Oh(t) = M₁(t) / (M₁_CALM + ε)              [Rayleigh criterion]
    Φ(t)  = γ·Φ(t-1) + max(0, −κ̄(t) − κ̄_CALM) [Prandtl criterion]
    Ξ(t)  = h₁_count(t) / Ξ_c
    DEF(t)= 1 − |cos(S̃(t), ΔS̃(t))|
        │
        ▼
Structural Pressurization Signature (SPS) — all six conditions joint
        │
        ▼
Regime transition warning + viscosity νs + preparation ratio PR
```

---

## Validated Scenarios — Version 3 (17 scenarios, 1985–2023)

| Scenario | ∫Φ | νs | f_K | Result |
|---|---|---|---|---|
| **GFC 2008** | 8.05 | 700.4 | 30.4% | ✅ DETECTED — ~95 days before Lehman |
| **Eurozone 2011** | 24.1 | 8.2 | 6.7% | ✅ DETECTED — ~1 week before ECB/IMF intervention |
| **EU Housing 2007** | 104.9 | 18.4 | 50.1% | ✅ DETECTED |
| **EM Crisis 2018** | 114.3 | 10.0 | 64.2% | ✅ DETECTED — highest f_K in dataset |
| **SVB 2023** | 182.9 | 29.5 | 32.0% | ✅ DETECTED — post-2020 validation |
| **China 2015** | 51.0 | 4.7 | 6.9% | ✅ DETECTED |
| **Commodities 2014** | 37.3 | 1.0 | 6.1% | ✅ DETECTED |
| **Global Equity 2015** | 24.1 | 3.0 | 4.4% | ✅ DETECTED |
| **COVID-19 2020** | 413.6† | — | 58.1% | ✅ CORRECT NEGATIVE — exogenous shock |
| **Rates 2022** | 14.7 | — | 8.1% | ✅ CORRECT NEGATIVE |
| **Brexit 2016** | 15.0 | — | 7.1% | ✅ CORRECT NEGATIVE |
| **Taper Tantrum 2013** | ~0 | NaN | 4.5% | ✅ CORRECT NEGATIVE |
| **Repo Crisis 2019** | ~0 | NaN | 53.9% | ✅ CORRECT NEGATIVE |
| **Volmageddon 2018** | 21.4 | — | 43.9% | ~ SPEED-LIMITED (sub-window event) |
| **Flash Crash 2010** | 34.0 | — | 43.9% | ~ SPEED-LIMITED |
| **LTCM 1998** | ~0 | NaN | 56.9% | · NON-DETECTION (Katashi-improdutive) |
| **Crash 1987** | weak | — | — | ~ PARTIAL SIGNAL |

> † COVID-19: ∫Φ accumulates *after* the shock, not before. The SPS was not satisfied pre-event — correct negative by structural timing, not post-hoc exclusion.

**Sensitivity: 8/8** for scenarios satisfying the SPS criterion. **Specificity: 5/5** — no false positives across five large drawdown events.

---

## Structural Pressurization Signature (SPS)

Formal detection criterion — Definition 3.1 of the v3 paper:

```
(i)   Oh(t)  > Oh_pre          — topological pressure above CALM ceiling
(ii)  Φ(t)   > Φ_c             — structural memory active
(iii) ΔΦ(t)  > 0               — memory accumulating, not dissipating
(iv)  η(t)   > η_ref            — dynamic rigidity above baseline
(v)   Ξ(t)   < Ξ_c             — structural diversity collapsing
(vi)  DEF(t) < DEF_q75(CALM)   — low angular divergence (directional coherence)
```

All thresholds are derived from the CALM ensemble of the same series — no external calibration required. The criterion is **prospective**: requires no prior knowledge of whether a crisis will occur, its timing, or its cause. It is **falsifiable**: SPS without subsequent disruption = false positive, must be reported.

---

## Installation

```bash
git clone https://github.com/aprimora-ai/Kappa-FIN.git
cd Kappa-FIN
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, Pandas, SciPy, GUDHI, NetworkX, yfinance, Matplotlib

---

## Quickstart

```bash
# Run all 17 scenarios (v3 engine)
python scripts/run_scenarios_v3.py

# Single scenario
python scripts/run_scenarios_v3.py gfc2008

# Comparative typology analysis
python scripts/run_typology_v3.py
```

### From Python

```python
from kappa_fin.engine_v3 import run_scenario

result = run_scenario(
    scenario_id="gfc2008",
    out="./results/v3_gfc2008"
)
# result contains: Oh, Phi, eta, Xi, DEF time series
# + phi_crossing, nu_s, PR, f_K, SPS duration
```

---

## Outputs

Each scenario run produces in `results/v3_<scenario>/`:

| File | Description |
|---|---|
| `kappa_v3_state.csv` | Full time series: Oh, Φ, η, Ξ, DEF, v_raw, Katashi flag |
| `kappa_v3_report.txt` | Crossing dates, SPS duration, νs, PR, CALM parameters |
| `kappa_v3_dashboard.png` | 4-panel dashboard: Oh(t), Φ(t), η(t)/Ξ(t), DEF(t) |
| `kappa_v3_viscosity.csv` | Derived metrics: νs, PR, f_K, τ_SPS per scenario |

Comparative analysis (`run_typology_v3.py`) produces in `results/typology_v3/`:

| File | Description |
|---|---|
| `kappa_v3_metrics.csv` | All 17 scenarios × all metrics |
| `kappa_v3_table.tex` | LaTeX table for paper |
| `kappa_v3_figure.png` | 4-panel comparative figure |

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `window` | 22 | Rolling window (trading days) |
| `k` | 5 | k-NN graph connectivity |
| `gamma` | 0.97 | Structural memory decay |
| `delta` | 0.08 | Damage sensitivity |
| `pre_q` | 0.968 | CALM quantile for Oh threshold |
| `dep` | spearman+LW(λ=0.05) | Correlation estimator |
| CALM | ensemble v4.6 + KCPT | Automated baseline identification |

**No per-scenario parameter tuning.** Identical parameters across all 17 scenarios.

---

## Repository Structure

```
kappa-fin/
├── kappa_fin/
│   ├── __init__.py
│   ├── engine.py          # v1/v2 engine (preserved for reference)
│   └── engine_v3.py       # v3 engine — CALM ensemble, DEF, νs, PR, SPS
├── scripts/
│   ├── run_scenarios_v3.py   # 17-scenario registry + executor (v3)
│   ├── run_typology_v3.py    # Comparative analysis across all scenarios
│   └── run_*.py              # Individual scenario scripts (v1/v2, preserved)
├── results/
│   ├── typology_v3/          # Comparative outputs (metrics, table, figure)
│   └── v3_*/                 # Per-scenario v3 outputs
├── paper/
│   ├── artigo_kappa_fin_en_v3.pdf   # Version 3 paper (English)
│   ├── artigo_kappa_fin_en_v2.pdf   # Version 2 (preserved)
│   └── figures/
├── docs/
│   └── method.md
├── tests/
├── requirements.txt
├── setup.py
├── CITATION.cff
└── LICENSE
```

---

## Academic Paper

> Ohio, David. *Topological Detection of Regime Transitions in Financial Markets: The Kappa-FIN Method*. Version 3 — Extended Validation (17 Scenarios, 1985–2023) with Prospective Detection Criterion. Zenodo, 2026. DOI: [10.5281/zenodo.18883821](https://doi.org/10.5281/zenodo.18883821)

Full paper: [`paper/artigo_kappa_fin_en_v3.pdf`](paper/artigo_kappa_fin_en_v3.pdf)

---

## Theoretical Background

Kappa-FIN is an empirical application of the **Law of Ohio** (Organized Hallucination In Optimization): information-processing systems losing access to external validation progressively optimize internal coherence at the expense of reality correspondence — a process detectable through topological invariants.

In financial markets: as price discovery degrades and assets decouple from fundamentals, the correlation network becomes self-referentially coherent — high rigidity, growing memory, collapsing topological diversity. The observable vector S(t) = (Oh, Φ, η, Ξ, DEF) tracks this migration continuously. The SPS is the joint signature of all five observables entering a pressurization regime simultaneously.

The method's attractor-tracking design supports a natural trajectory toward **probabilistic regime forecasting**: as the scenario corpus grows, the binary SPS warning becomes a nearest-neighbor query in observable space — what fraction of historical trajectories with similar S(t) underwent a regime transition within L days?

---

## Citation

```bibtex
@software{ohio2026kappafin,
  author    = {Ohio, David},
  title     = {Kappa-FIN: Topological Early Warning System for Financial Market Crises},
  version   = {3.0.0},
  year      = {2026},
  doi       = {10.5281/zenodo.18883821},
  url       = {https://github.com/aprimora-ai/Kappa-FIN},
  license   = {CC BY 4.0}
}
```

---

## Related Work

- **[Kappa-Method](https://github.com/aprimora-ai/Kappa-Method)** — Theoretical foundation: Oh, Φ, η, Ξ, DEF observables and the Law of Ohio
- **[Kappa-LLM / Kappa-Attention-Regimes](https://github.com/aprimora-ai/Kappa-Attention-Regimes)** — Hallucination detection in LLMs (AUC up to 94.2% on Phi-3, Mistral, Llama)
- **[Kappa-Education](https://github.com/aprimora-ai/Kappa-Education)** — Student dropout prediction via topological engagement dynamics (OULAD dataset)

---

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

> Ohio, David. *Kappa-FIN: Topological Early Warning System for Financial Market Crises*. 2026. https://github.com/aprimora-ai/Kappa-FIN
