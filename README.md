# Kappa-FIN

**Topological Early Warning System for Financial Market Crises**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18883821.svg)](https://doi.org/10.5281/zenodo.18883821)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Part of Kappa-Method](https://img.shields.io/badge/Kappa--Method-ecosystem-brightgreen)](https://github.com/aprimora-ai/Kappa-Method)

Kappa-FIN applies the **Kappa Method** — persistent homology H₁ + Forman-Ricci curvature on rolling correlation networks — to detect systemic risk precursors in financial markets weeks to months before crises materialize.

**Author:** David Ohio · [odavidohio@gmail.com](mailto:odavidohio@gmail.com) · Independent Researcher  
**Repository:** [github.com/aprimora-ai/Kappa-FIN](https://github.com/aprimora-ai/Kappa-FIN)

---

## Core Hypothesis

Financial crises do not emerge from chaos — they emerge from **organized topological coherence**.

The Kappa Method quantifies this via two complementary criteria:

| Criterion | Observable | Meaning |
|---|---|---|
| **Rayleigh** | Ohio Number `Oh(t) = Ξ(t) / Ξ_c` | Structural coupling relative to the calm baseline |
| **Prandtl** | Damage Integral `φ(t)` | Accumulated excess exposure above the pre-crossing threshold |

A **dual crossing** — `Oh > 1` followed by `φ > φ_c` — constitutes a confirmed structural regime change (not a false alarm).

---

## Method Overview

```
Market prices (yfinance)
        │
        ▼
Log-returns  ──► Rolling 22-day window
        │
        ▼
Spearman correlation matrix  ──► Ledoit shrinkage ──► Distance matrix
        │
        ▼
k-NN graph (k=5)
    ├── Forman-Ricci curvature  ──► η(t) = 1 / (|κ| + η_floor)
    └── Rips complex (GUDHI)
            │
            ▼
    H₁ persistent homology ──► entropy, dominance, β₁ count, mass
            │
            ▼
    v_raw = entropy × (1 − dominance)
            │
            ▼
CALM identification (automated scan, KCPT penalty)
        │
        ▼
    Ξ(t) = η_ref / η(t)     ──► Oh(t) = Ξ(t) / Ξ_c       [Rayleigh]
    φ(t) = γ·φ(t-1) + max(Oh − Oh_pre − δ, 0)             [Prandtl]
        │
        ▼
Dual crossing detection (sensitivity + confirmation persistence)
        │
        ▼
Analysis report + phase plots + warming indicators
```

---

## Validated Crisis Scenarios

| Crisis | Period | Tickers | Result |
|---|---|---|---|
| **GFC 2008** | 2005–2010 | 14 ETFs¹ | ✅ φ crossing confirmed: **2008-07-17** (~60 days before Lehman) |
| **COVID-19 2020** | 2018–2021 | 14 ETFs | ✅ **Correct negative** — no structural precursor detected (exogenous shock) |
| **1987 Crash** | 1985–1989 | Equity ETFs/proxies | ⚠️ No crossing detected (data universe limitation) |
| **Dot-com 2000** | 1999–2003 | SPY, QQQ, DIA, GLD, TLT | ✅ φ crossing confirmed: 2002-08-21 (slow-motion structural collapse) |
| **Volmageddon 2018** | 2017–2019 | Equities + VIX proxies | ✅ Oh crossing confirmed: 2019-09-25 |
| **Euro/Taper 2011** | 2009–2015 | Multi-asset | 🔄 In progress |
| **Rates 2022** | 2021–2023 | Bonds + equities | 🔄 In progress |

> **Note on COVID-19:** The absence of a signal is the expected and scientifically correct result. The Kappa Method detects *endogenous* structural deterioration — the gradual loss of adaptive capacity that precedes self-generated crises. The COVID-19 crash was an exogenous shock with no internal structural precursor. This discriminative capacity (distinguishing endogenous collapse from exogenous shock) is absent from conventional risk metrics.

---

> ¹ HYG (iShares HY Corporate Bond ETF) excluded: launched April 2007, no data for the 2006 pre-crisis window.
> USO (US Oil Fund ETF) excluded: late-start NaN rows shift `dates[0]` to May 2006, preventing the 504-day
> CALM scan from finding a valid window starting in January 2006. Both are data availability constraints,
> not methodological choices. The ticker list used in the study is declared in `scripts/run_gfc2008.py`.

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
# GFC 2008 (canonical validation)
python scripts/run_gfc2008.py --out results/gfc2008

# COVID 2020
python scripts/run_covid2020.py

# Fed rate hike cycle 2022
python scripts/run_rates2022.py
```

### From Python

```python
from kappa_fin import Config, run

cfg = Config(
    tickers=["SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV",
             "TLT", "IEF", "LQD", "HYG", "GLD", "USO", "DIA"],
    start="2006-01-01",
    end="2010-07-01",
    calm_policy="scan",
    calm_search_to="2007-06-30",
    calm_length_days=504,
    delta=0.08,
    gamma=0.985,
    pre_q=0.968,
    out="./results/gfc2008",
)

run(cfg)
```

---

## Outputs

Each run produces in `--out` directory:

| File | Description |
|---|---|
| `kappa_fin_state.csv` | Full time series: η, Ξ, Oh, φ, H₁ observables |
| `analysis_report.txt` | Crossing dates, warming indicators, calibration params |
| `observables.png` | H₁ entropy, dominance, v_raw, mean correlation |
| `ohio_number.png` | Oh(t) with CALM end and threshold lines |
| `damage_phi.png` | φ(t) with φ_c threshold |

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `window` | 22 | Rolling window (trading days) |
| `k` | 5 | k-NN graph connectivity |
| `calm_length_days` | 504 | CALM segment length (~2 trading years) |
| `calm_search_to` | — | Search CALM before this date |
| `delta` | 0.08 | Damage sensitivity |
| `gamma` | 0.985 | Structural memory decay — γ ∈ [0.98, 0.99] |
| `pre_q` | 0.968 | Quantile for Oh baseline |
| `oh_persist_sens` | 2 | Sensitivity persistence (days) |
| `oh_persist_confirm` | 5 | Confirmation persistence (days) |

---

## CALM Identification (v4.6)

The baseline period ("CALM") is identified automatically via a scanning algorithm that minimizes structural volatility:

```
calm_score = std(η) + std(mean_corr) + mean|Δη| + mean|Δmean_corr| + 0.25·mean(v_raw)
           + penalty_knee_phi_tail    # v4.6: penalizes internal phase transitions
```

The **knee-phi-tail penalty** (KCPT) prevents the algorithm from selecting a baseline window that itself contains an incipient crisis, a subtle failure mode that affected earlier versions.

---

## Theoretical Background

Kappa-FIN is an empirical application of the **Law of Ohio** (Organized Hallucination In Optimization): information-processing systems losing access to external validation optimize internal coherence at the expense of reality correspondence.

In financial markets: as price discovery degrades (systemic stress, correlated selling), the correlation network transitions from sparse/heterogeneous topology to a tightly organized, high-coherence configuration — precisely the signature H₁ persistent homology detects.

This generalizes across domains:
- **LLMs:** hallucinating models show 15–18% higher topological coherence than factual responses → [Kappa-LLM](https://github.com/aprimora-ai/Kappa-Attention-Regimes)
- **Education:** dropout students show φ-crossing weeks before disengagement → [Kappa-EDU](https://github.com/aprimora-ai/Kappa-Education)
- **Financial markets:** pre-crisis periods show organized H₁ mass growth → this repository

---

## Repository Structure

```
kappa-fin/
├── kappa_fin/
│   ├── __init__.py          # Package metadata
│   └── engine.py            # Core pipeline (Config, run, all components)
├── scripts/
│   ├── run_gfc2008.py       # 2008 Global Financial Crisis
│   ├── run_covid2020.py     # COVID-19 crash
│   ├── run_dotcom2000.py    # Dot-com bubble
│   ├── run_rates2022.py     # Fed rate hike / bond crisis 2022
│   ├── save_study_data.py   # Download & version-lock price data
│   └── verify_data.py       # Verify SHA-256 checksums
├── notebooks/               # Jupyter demo notebooks (coming soon)
├── docs/
│   └── method.md            # Mathematical documentation
├── data/results/            # Curated result CSVs (versioned)
├── paper/figures/           # Publication-quality figures
├── tests/                   # Unit tests
├── requirements.txt
├── setup.py
├── setup_data.py            # First-time data download helper
├── CITATION.cff
└── LICENSE                  # CC BY 4.0
```

---

## Academic Papers

Formal peer-review-ready write-ups of the method and its financial validation are available in the `paper/` directory:

| Language | File |
|---|---|
| English | [`paper/artigo_kappa_fin_en_v3.pdf`](paper/artigo_kappa_fin_en_v3.pdf) |

The papers document: (1) full mathematical formalism aligned with the Kappa Method README; (2) GFC 2008 case-study results with dual-crossing timeline; (3) COVID-19 correct-negative result; (4) reproducibility block with exact parameters and SHA-256 checksums.

---

## Citation

```bibtex
@software{ohio2026kappafin,
  author    = {Ohio, David},
  title     = {Kappa-FIN: Topological Early Warning System for Financial Market Crises},
  year      = {2026},
  doi       = {10.5281/zenodo.18883821},
  url       = {https://github.com/aprimora-ai/Kappa-FIN},
  license   = {CC BY 4.0}
}
```

---

## Related Work

- **[Kappa-Method](https://github.com/aprimora-ai/Kappa-Method)** — Theoretical foundation: Oh, Φ, η, Ξ, DEF observables and the Law of Ohio
- **[Kappa-LLM / Kappa-Attention-Regimes](https://github.com/aprimora-ai/Kappa-Attention-Regimes)** — Hallucination detection in large language models (AUC up to 94.2% on Phi-3)
- **[Kappa-EDU](https://github.com/aprimora-ai/Kappa-Education)** — Student dropout prediction via topological engagement dynamics (OULAD dataset)

---

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

You are free to use, share, and adapt this work for any purpose, including commercial use, provided you give appropriate credit:

> Ohio, David. *Kappa-FIN: Topological Early Warning System for Financial Market Crises*. 2026. https://github.com/aprimora-ai/Kappa-FIN
