#!/usr/bin/env python3
"""
Kappa-FIN v3 — Análise Comparativa de Cenários Históricos
==========================================================

Lê os kappa_v3_state.csv de todos os cenários e produz:
  1. Tabela comparativa de observáveis do Kappa Method por cenário
  2. Figura com 4 painéis: trajetórias Φ(t), ∫Φ, ν_s vs PR, regimes
  3. LaTeX table para o paper

Observáveis reportados — todos derivados do formalismo do Kappa Method:
  ∫Φ   : integral da memória estrutural acumulada
  ν_s  : viscosidade estrutural = τ_K × η_K / (Ξ_K + floor)
  PR   : Razão de Preparação = Φ(t*-ε) / Φ(t*)
           PR → 1: dano pré-existente ao evento (assinatura endógena)
           PR → 0: dano aparece no/após o evento (assinatura exógena)
  DEF_K: divergência angular média durante Katashi
  f_K  : fração do período em Katashi

Nota metodológica: nenhum rótulo externo (tipo A/B/C) é usado.
A classificação emergente, se existir, resulta dos próprios observáveis.

Autor: David Ohio <odavidohio@gmail.com>
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

ROOT    = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent

# ── Registro de cenários ────────────────────────────────────────────────────
# Apenas metadados neutros: data do evento de referência e label.
# Cores: paleta qualitativa sequencial — sem semântica de grupo.
_COLORS = [
    "#1a4a7a", "#0277bd", "#006064", "#1b5e20", "#bf360c",
    "#4527a0", "#00695c", "#2196f3", "#00bcd4", "#e65100",
    "#f57f17", "#ff8f00", "#d84315", "#6d4c41", "#c62828",
    "#ad1457", "#4e342e", "#78909c", "#546e7a", "#37474f",
]

REGISTRY = {
    "gfc2008":               {"event": "2008-09-15", "label": "GFC 2008"},
    "dotcom2000":            {"event": "2000-03-10", "label": "Dot-com 2000"},
    "euro2011":              {"event": "2011-07-05", "label": "Euro 2011"},
    "ltcm1998":              {"event": "1998-08-17", "label": "LTCM 1998"},
    "china2015":             {"event": "2015-08-24", "label": "China 2015"},
    "eu_housing2007":        {"event": "2007-06-01", "label": "EU Housing 2007"},
    "credit_bubble2007":     {"event": "2007-07-01", "label": "Credit Bubble 2007"},
    "global_2008_multiasset":{"event": "2008-09-15", "label": "GFC Multi-Asset"},
    "global_equity_2015":    {"event": "2015-08-24", "label": "Global Equity 2015"},
    "crash1987":             {"event": "1987-10-19", "label": "Crash 1987"},
    "taper_tantrum2013":     {"event": "2013-05-22", "label": "Taper 2013"},
    "repo_crisis2019":       {"event": "2019-09-17", "label": "Repo 2019"},
    "commodities_2014":      {"event": "2014-06-01", "label": "Commodities 2014"},
    "em_crisis_2018":        {"event": "2018-08-01", "label": "EM 2018"},
    "covid2020":             {"event": "2020-03-20", "label": "COVID 2020"},
    "rates2022":             {"event": "2022-06-16", "label": "Juros 2022"},
    "volmageddon2018":       {"event": "2018-02-05", "label": "Volmageddon 2018"},
    "flashcrash2010":        {"event": "2010-05-06", "label": "Flash Crash 2010"},
    "brexit2016":            {"event": "2016-06-23", "label": "Brexit 2016"},
    "svb2023":               {"event": "2023-03-10", "label": "SVB 2023"},
}
for i, meta in enumerate(REGISTRY.values()):
    meta["color"] = _COLORS[i % len(_COLORS)]

PHI_ACTIVE_THR = 1e-4
PR_EPSILON_DAYS = 5   # janela ε para PR (dias úteis antes do evento)


def find_csv(name: str) -> Path | None:
    for d in [f"v3_{name}", name]:
        for base in [ROOT, SCRIPTS]:
            for p in [
                base / "results" / d / "kappa_v3_state.csv",
                base / f"out_{name}" / "kappa_v3_state.csv",
            ]:
                if p.exists():
                    return p
    return None


def compute_metrics(df: pd.DataFrame, event_date: str) -> dict:
    """
    Computa observáveis do Kappa Method para um cenário.

    PR (Razão de Preparação) — formalizada em Ohio (2026):
        PR = Φ_cum(t* - ε) / Φ_cum(t*)
        onde Φ_cum é a integral acumulada de Φ até a data t.
        PR → 1: preparação estrutural pré-existente (assinatura endógena)
        PR → 0: dano emerge no evento ou após (assinatura exógena)
        PR = NaN: Φ_cum(t*) ≈ 0 (sem memória acumulada)

    ν_s (viscosidade estrutural):
        ν_s = τ_K × η_K / (Ξ_K + floor)
        Definido apenas quando ∫Φ ≥ 0.01 (memória não-trivial).
    """
    phi = df["phi"].values
    xi  = df["Xi"].values  if "Xi"  in df.columns else np.ones(len(df))
    eta = df["eta"].values
    DEF = df["DEF"].values if "DEF" in df.columns else np.zeros(len(df))
    reg = df["regime"].values if "regime" in df.columns else np.full(len(df), "?")
    ev  = pd.Timestamp(event_date)

    # ── Observáveis básicos ──────────────────────────────────────────────────
    phi_integral = float(phi.sum())
    phi_max      = float(phi.max())

    # τ_Φ: maior run contínuo com Φ ativo
    phi_active = phi > PHI_ACTIVE_THR
    runs = []; cur = 0
    for s in phi_active:
        if s: cur += 1
        else:
            if cur > 0: runs.append(cur); cur = 0
    if cur > 0: runs.append(cur)
    tau_phi_max = float(max(runs)) if runs else 0.0

    # ── Regime Katashi ───────────────────────────────────────────────────────
    kat_mask = reg == "Katashi"
    runs_k = []; cur = 0
    for s in kat_mask:
        if s: cur += 1
        else:
            if cur > 0: runs_k.append(cur); cur = 0
    if cur > 0: runs_k.append(cur)
    tau_K_max = float(max(runs_k)) if runs_k else 0.0
    eta_K = float(eta[kat_mask].mean()) if kat_mask.sum() > 0 else 0.0
    xi_K  = float(xi[kat_mask].mean())  if kat_mask.sum() > 0 else 0.0

    # ν_s: só definido com memória acumulada não-trivial
    xi_floor = max(
        float(np.quantile(xi[xi > 0], 0.05)) if (xi > 0).sum() > 5 else 1e-9,
        1e-9
    )
    nu_s_raw = tau_K_max * eta_K / (xi_K + xi_floor)
    nu_s = nu_s_raw if phi_integral >= 0.01 else float("nan")

    # DEF médio durante Katashi
    def_K  = float(DEF[kat_mask].mean()) if kat_mask.sum() > 0 else 0.0
    frac_K = float(kat_mask.mean())

    # Frações por regime
    frac_N = float((reg == "Nagare").mean())
    frac_U = float((reg == "Utsuroi").mean())

    # ── PR: Razão de Preparação ──────────────────────────────────────────────
    # Φ acumulado até t* e até t* - ε (onde ε = PR_EPSILON_DAYS dias úteis)
    phi_cum = np.cumsum(phi)
    idx_ev = np.searchsorted(df.index, ev, side="right") - 1
    idx_ev = max(0, min(idx_ev, len(df) - 1))

    phi_at_event = phi_cum[idx_ev]
    idx_pre = max(0, idx_ev - PR_EPSILON_DAYS)
    phi_pre_event = phi_cum[idx_pre]

    if phi_at_event >= 0.01:
        PR = float(phi_pre_event / phi_at_event)
    else:
        PR = float("nan")  # sem memória — PR indefinido

    return {
        "phi_integral": phi_integral,
        "phi_max":      phi_max,
        "tau_phi_max":  tau_phi_max,
        "tau_K_max":    tau_K_max,
        "eta_K":        eta_K,
        "xi_K":         xi_K,
        "nu_s":         nu_s,
        "def_K":        def_K,
        "frac_K":       frac_K,
        "frac_N":       frac_N,
        "frac_U":       frac_U,
        "PR":           PR,
    }


def main():
    out_dir = ROOT / "results" / "typology_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("  Kappa-FIN v3 — Análise Comparativa de Cenários Históricos")
    print("  David Ohio · odavidohio@gmail.com")
    print("=" * 76)
    print(f"  {'Cenário':<30}  {'∫Φ':>9}  {'ν_s':>7}  {'PR':>6}  {'f_K':>5}  DEF_K")
    print(f"  {'─'*30}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*5}")

    loaded = []
    for name, meta in REGISTRY.items():
        csv = find_csv(name)
        if csv is None:
            print(f"  ⚠  {name:<30} — não encontrado")
            continue
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        m  = compute_metrics(df, meta["event"])

        flag    = "✓" if m["phi_integral"] > 10 else "~" if m["phi_integral"] > 0.1 else "·"
        nu_str  = f"{m['nu_s']:>7.2f}"  if np.isfinite(m["nu_s"]) else "    NaN"
        pr_str  = f"{m['PR']:>6.3f}"    if np.isfinite(m["PR"])   else "   NaN"
        fk_str  = f"{m['frac_K']*100:>4.1f}%"

        print(f"  {flag}  {meta['label']:<30}  "
              f"∫Φ={m['phi_integral']:>8.3f}  ν_s={nu_str}  "
              f"PR={pr_str}  f_K={fk_str}  DEF={m['def_K']:.3f}")
        loaded.append({"name": name, "meta": meta, "df": df, "m": m})

    if not loaded:
        print("\n  ❌ Nenhum CSV encontrado. Execute run_scenarios_v3.py primeiro.")
        return

    # ── Tabela CSV ────────────────────────────────────────────────────────────
    rows = [{"cenario": r["meta"]["label"], **r["m"]} for r in loaded]
    tbl  = pd.DataFrame(rows).sort_values("phi_integral", ascending=False)
    tbl.to_csv(out_dir / "kappa_v3_metrics.csv", index=False)
    print(f"\n  ✅ Tabela: {out_dir / 'kappa_v3_metrics.csv'}")

    # ── Sumário geral ─────────────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print("  SUMÁRIO — observáveis do Kappa Method (todos os cenários):\n")
    all_phi = [r["m"]["phi_integral"] for r in loaded]
    all_nus = [r["m"]["nu_s"] for r in loaded if np.isfinite(r["m"]["nu_s"])]
    all_pr  = [r["m"]["PR"]  for r in loaded if np.isfinite(r["m"]["PR"])]
    all_fk  = [r["m"]["frac_K"] for r in loaded]

    print(f"  ∫Φ    :  μ={np.mean(all_phi):.2f}  σ={np.std(all_phi):.2f}  "
          f"min={np.min(all_phi):.3f}  max={np.max(all_phi):.2f}")
    if all_nus:
        print(f"  ν_s   :  μ={np.mean(all_nus):.2f}  σ={np.std(all_nus):.2f}  "
              f"max={np.max(all_nus):.2f}  (N={len(all_nus)}/{len(loaded)})")
    if all_pr:
        print(f"  PR    :  μ={np.mean(all_pr):.3f}  σ={np.std(all_pr):.3f}  "
              f"min={np.min(all_pr):.3f}  max={np.max(all_pr):.3f}  (N={len(all_pr)}/{len(loaded)})")
    print(f"  f_K   :  μ={np.mean(all_fk)*100:.1f}%  "
          f"min={np.min(all_fk)*100:.1f}%  max={np.max(all_fk)*100:.1f}%")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Cenário & $\int\Phi\,dt$ & $\nu_s$ & PR & $f_K$ & "
        r"$\overline{\mathrm{DEF}}_K$ \\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        nu_tex = f"{row['nu_s']:.2f}" if np.isfinite(row["nu_s"]) else r"---"
        pr_tex = f"{row['PR']:.3f}"   if np.isfinite(row["PR"])   else r"---"
        latex.append(
            f"  {row['cenario']} & {row['phi_integral']:.2f} & "
            f"{nu_tex} & {pr_tex} & "
            f"{row['frac_K']*100:.1f}\\% & {row['def_K']:.3f} \\\\"
        )
    latex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Observ\'aveis do Kappa Method por cen\'ario hist\'orico "
        r"(Kappa-FIN v3). "
        r"$\int\Phi\,dt$: mem\'oria estrutural acumulada; "
        r"$\nu_s = \tau_K \cdot \eta_K / \Xi_K$: viscosidade estrutural; "
        r"PR $= \Phi_{\text{cum}}(t^*-\varepsilon)/\Phi_{\text{cum}}(t^*)$: "
        r"raz\~ao de prepara\c{c}\~ao ($\varepsilon=5$ dias); "
        r"$f_K$: fra\c{c}\~ao do per\'iodo em regime Katashi; "
        r"$\overline{\mathrm{DEF}}_K$: diverg\^encia angular m\'edia em Katashi.}",
        r"\label{tab:kappa_scenarios}",
        r"\end{table}",
    ]
    (out_dir / "kappa_v3_table.tex").write_text("\n".join(latex), encoding="utf-8")
    print(f"  ✅ LaTeX: {out_dir / 'kappa_v3_table.tex'}")

    _plot_scenarios(loaded, out_dir)


def _plot_scenarios(loaded: list, out_dir: Path):
    """
    4 painéis — todos os observáveis emergem dos dados, sem rótulos externos.

    P1: Trajetórias Φ(t) normalizadas, alinhadas ao evento de referência
    P2: ∫Φ por cenário (barras horizontais, ordenadas)
    P3: ν_s vs PR — espaço bidimensional de preparação estrutural
    P4: Composição de regimes por cenário (Nagare / Utsuroi / Katashi)
    """
    fig = plt.figure(figsize=(22, 17))
    fig.patch.set_facecolor("#f5f6fa")
    gs = gridspec.GridSpec(2, 2, hspace=0.44, wspace=0.30,
                           left=0.05, right=0.97, top=0.93, bottom=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("white")
        ax.grid(alpha=0.18)

    # ── P1: Trajetórias Φ(t) normalizadas ────────────────────────────────────
    for r in loaded:
        phi  = r["df"]["phi"]
        phi_n = (phi - phi.min()) / (phi.max() - phi.min() + 1e-12)
        ev   = pd.Timestamp(r["meta"]["event"])
        days = (r["df"].index - ev).days / 365.0
        ax1.plot(days, phi_n.values,
                 color=r["meta"]["color"], lw=1.6, alpha=0.80)

    ax1.axvline(0, color="#c0392b", lw=2.0, linestyle="--", alpha=0.85,
                label="Evento de referência (t=0)")
    ax1.axvspan(-3, 0, alpha=0.03, color="#c0392b")
    ax1.legend(fontsize=9.5, loc="upper left")
    ax1.set_xlabel("Anos relativos ao evento de referência", fontsize=11)
    ax1.set_ylabel("Φ(t) normalizado [0, 1]", fontsize=11)
    ax1.set_xlim([-3.5, 2.5])
    ax1.set_title(
        "Trajetórias Φ(t) — Memória Estrutural por Cenário\n"
        "Alinhadas à data do evento de referência (t = 0)",
        fontsize=11, fontweight="bold")

    # ── P2: ∫Φ por cenário ───────────────────────────────────────────────────
    sorted_r = sorted(loaded, key=lambda r: r["m"]["phi_integral"], reverse=True)
    names_s   = [r["meta"]["label"]              for r in sorted_r]
    integrals = [max(r["m"]["phi_integral"], 1e-4) for r in sorted_r]
    colors_s  = [r["meta"]["color"]              for r in sorted_r]

    y_pos = range(len(names_s))
    ax2.barh(list(y_pos), [np.log10(v) for v in integrals],
             color=colors_s, height=0.65, edgecolor="white", alpha=0.9)
    ax2.axvline(np.log10(10), color="#888", lw=1.2, linestyle=":",
                alpha=0.7, label="∫Φ = 10")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(names_s, fontsize=8.5)
    for i, val in enumerate(integrals):
        ax2.text(np.log10(val) + 0.04, i, f"{val:.1f}",
                 va="center", fontsize=8, color=colors_s[i], fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlabel("log₁₀(∫Φ dt) — memória estrutural acumulada", fontsize=11)
    ax2.set_title(
        "Integral de Φ(t) por Cenário\n"
        "Ordenado por magnitude de memória estrutural",
        fontsize=11, fontweight="bold")

    # ── P3: ν_s vs PR ────────────────────────────────────────────────────────
    # Espaço bidimensional: eixo x = PR (preparação), eixo y = ν_s (viscosidade)
    # Quadrantes emergem dos dados — sem rótulos impostos externamente.
    for r in loaded:
        pr  = r["m"]["PR"]
        nu  = r["m"]["nu_s"]
        if not (np.isfinite(pr) and np.isfinite(nu)):
            continue
        ax3.scatter(pr, nu, color=r["meta"]["color"], s=150, zorder=5,
                    edgecolors="#333", lw=0.7)
        ax3.annotate(r["meta"]["label"].split(" ")[0], (pr, nu),
                     fontsize=7.5, ha="left", va="bottom",
                     xytext=(4, 3), textcoords="offset points",
                     color=r["meta"]["color"])

    # Linhas de referência medianas — emergem dos dados
    pr_vals  = [r["m"]["PR"]   for r in loaded if np.isfinite(r["m"]["PR"])  and np.isfinite(r["m"]["nu_s"])]
    nu_vals  = [r["m"]["nu_s"] for r in loaded if np.isfinite(r["m"]["PR"])  and np.isfinite(r["m"]["nu_s"])]
    if pr_vals:
        ax3.axvline(np.median(pr_vals), color="#aaa", lw=1.0, linestyle="--", alpha=0.6)
        ax3.axhline(np.median(nu_vals), color="#aaa", lw=1.0, linestyle="--", alpha=0.6)

    ax3.set_xlabel("PR = Φ_cum(t*−ε) / Φ_cum(t*)  [Razão de Preparação]", fontsize=10)
    ax3.set_ylabel("ν_s = τ_K × η_K / Ξ_K  [Viscosidade Estrutural]", fontsize=10)
    ax3.set_title(
        "Espaço de Preparação Estrutural: ν_s × PR\n"
        "PR → 1: dano pré-existente   PR → 0: dano emerge no evento",
        fontsize=10.5, fontweight="bold")
    ax3.set_yscale("symlog", linthresh=1.0)
    ax3.set_xlim([-0.05, 1.05])

    # ── P4: Composição de regimes por cenário ────────────────────────────────
    sorted_fk = sorted(loaded, key=lambda r: r["m"]["frac_K"], reverse=True)
    names_k   = [r["meta"]["label"] for r in sorted_fk]
    frac_N    = [r["m"]["frac_N"]   for r in sorted_fk]
    frac_U    = [r["m"]["frac_U"]   for r in sorted_fk]
    frac_K    = [r["m"]["frac_K"]   for r in sorted_fk]

    y4 = range(len(names_k))
    ax4.barh(list(y4), frac_N, height=0.65,
             color="#2e7d32", alpha=0.80, label="Nagare (fluxo)")
    ax4.barh(list(y4), frac_U, height=0.65, left=frac_N,
             color="#f9a825", alpha=0.80, label="Utsuroi (transição)")
    ax4.barh(list(y4), frac_K, height=0.65,
             left=[n + u for n, u in zip(frac_N, frac_U)],
             color="#c62828", alpha=0.80, label="Katashi (colapso)")
    ax4.set_yticks(list(y4))
    ax4.set_yticklabels(names_k, fontsize=8.5)
    ax4.set_xlim([0, 1])
    ax4.set_xlabel("Fração do período de análise", fontsize=11)
    ax4.set_title(
        "Composição de Regimes por Cenário\n"
        "Nagare / Utsuroi / Katashi — ordenado por f_K decrescente",
        fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9.5, loc="lower right")

    fig.suptitle(
        "Kappa-FIN v3 — Observáveis do Kappa Method por Cenário Histórico\n"
        "S(t) = (Oh, Φ, η, Ξ, DEF)  ·  d_ij = √(2(1−C_ij))  ·  20 cenários  ·  1987–2024",
        fontsize=13, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.01,
        "David Ohio · odavidohio@gmail.com · "
        "Kappa-FIN v3 · DOI: 10.5281/zenodo.18883821",
        ha="center", fontsize=9, color="#888")

    out_path = out_dir / "kappa_v3_figure.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figura: {out_path}")


if __name__ == "__main__":
    main()