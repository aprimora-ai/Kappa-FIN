#!/usr/bin/env python3
"""
Kappa-FIN v3 — Registro Central de Cenários Expandidos
=======================================================

12 cenários históricos + 8 novos  =  20 cenários totais
Cobertura: 1987-2024, US + EU + EM + FX + Commodities

Autor: David Ohio <odavidohio@gmail.com>
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from kappa_fin.engine_v3 import ConfigV3, ASSET_PRESETS, run

def _out(name: str) -> str:
    """Caminho absoluto para o output de um cenário, sempre em ROOT/results/."""
    return str(ROOT / "results" / name)

# ══════════════════════════════════════════════════════════════════════════════
# PARÂMETROS GLOBAIS v3 — alinhados ao README
# ══════════════════════════════════════════════════════════════════════════════

V3_BASE = dict(
    window=22,           # janela rolante (dias úteis)
    k=5,                 # vizinhos kNN
    dep_method="spearman",
    shrink_lambda=0.05,  # Ledoit-Wolf
    dist_alpha=1.0,      # distância angular pura: sqrt(2(1-C))
    gamma=0.97,          # dissipação Φ
    delta=0.08,          # margem Oh→Φ
    pre_q=0.968,         # quantil Oh_pre
    phi_q=0.99,
    phi_persist=3,
    calm_length_days=504,
    calm_step_days=14,
    calm_topn=5,
    calm_knee_weight=1.0,
    calm_knee_tail_frac=0.25,
    calm_knee_ratio=1.50,
    calm_knee_slope_ann=0.0,
    calm_knee_mix=0.5,
    oh_persist_sens=2,
    oh_persist_confirm=5,
    eval_after_calm=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE CENÁRIOS
# Tipo A: bolha endógena (Φ preditivo)
# Tipo B: degradação estrutural (Φ = indicador de saúde)
# Tipo C: exógena/mecânica (Φ ≈ 0, correto negativo)
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {

    # ── TIPO A: BOLHAS ENDÓGENAS ──────────────────────────────────────────────

    "gfc2008": dict(
        title="GFC 2008 — Crise Financeira Global",
        type="A",
        tickers=[
            "SPY", "QQQ", "XLF", "XLE", "XLK",
            "JPM", "BAC", "C", "GS", "MS",
            "TLT", "LQD", "GLD",   # LQD (investment grade, desde 2002) substitui HYG (lançado abr/2007)
        ],
        start="2004-01-01", end="2010-12-31",
        calm_search_to="2006-12-31",
        out=_out("v3_gfc2008"),
    ),

    "dotcom2000": dict(
        title="Dot-com 2000 — Bolha Tecnológica",
        type="A",
        tickers=[
            "SPY", "QQQ", "XLK",
            "MSFT", "INTC", "CSCO", "ORCL",
            "TLT", "GLD",
        ],
        start="1997-01-01", end="2003-12-31",
        calm_search_to="1999-01-01",
        out=_out("v3_dotcom2000"),
    ),

    "euro2011": dict(
        title="Crise Europeia 2011 — Dívida Soberana",
        type="A",
        tickers=[
            "EWG", "EWI", "EWP", "EWQ", "EWU",   # Europa
            "SPY", "TLT", "GLD", "HYG", "EEM",
        ],
        start="2009-01-01", end="2014-12-31",
        calm_search_to="2010-06-30",
        out=_out("v3_euro2011"),
    ),

    "ltcm1998": dict(
        title="LTCM 1998 — Crise da Rússia/LTCM",
        type="A",
        tickers=[
            "SPY", "TLT", "GLD",
            "EWJ", "EEM",
            "JPM", "BAC", "C",
        ],
        start="1996-01-01", end="2000-06-30",
        calm_search_to="1997-12-31",
        out=_out("v3_ltcm1998"),
    ),

    "china2015": dict(
        title="China/EM 2015 — Black Monday China",
        type="A",
        tickers=[
            "FXI", "EEM", "EWZ", "EWY",
            "USO", "GLD", "XLE", "XLB",
            "SPY", "EFA",
        ],
        start="2013-01-01", end="2017-06-30",
        calm_search_to="2014-12-31",
        out=_out("v3_china2015"),
    ),

    # NOVO: Bolha imobiliária espanhola / europeia
    "eu_housing2007": dict(
        title="Bolha Imobiliária Europeia 2007",
        type="A",
        tickers=[
            "EWG", "EWP", "EWI", "EWU", "EWQ",
            "SPY", "TLT", "GLD", "HYG", "EEM",
        ],
        start="2004-01-01", end="2009-12-31",
        calm_search_to="2005-12-31",
        out=_out("v3_eu_housing2007"),
    ),

    # NOVO: Bolha de crédito corporativo 2006-2007
    "credit_bubble2007": dict(
        title="Bolha de Crédito Corporativo 2007",
        type="A",
        tickers=[
            "HYG", "LQD", "MBB",            # crédito
            "XLF", "JPM", "BAC", "C",        # financeiras
            "TLT", "SPY", "GLD",
        ],
        start="2004-01-01", end="2009-06-30",
        calm_search_to="2005-06-30",
        out=_out("v3_credit_bubble2007"),
    ),

    # ── TIPO B: DEGRADAÇÃO ESTRUTURAL ─────────────────────────────────────────

    "crash1987": dict(
        title="Crash de 1987 — Black Monday",
        type="B",
        tickers=[
            "SPY",   # proxy: SPDR fund existe desde 1993, usamos dado histórico
            "TLT", "GLD",
            "XLF", "XLK", "XLI",
        ],
        start="1985-01-01", end="1990-12-31",
        calm_search_to="1986-12-31",
        out=_out("v3_crash1987"),
    ),

    "taper_tantrum2013": dict(
        title="Taper Tantrum 2013 — Fed Bernanke",
        type="B",
        tickers=[
            "TLT", "IEF", "SHY",   # bonds
            "HYG", "LQD", "EMB",   # crédito/EM
            "XLU", "VNQ",          # sensíveis a juros
            "EEM", "SPY",
        ],
        start="2012-01-01", end="2015-12-31",
        calm_search_to="2013-01-31",
        out=_out("v3_taper_tantrum2013"),
    ),

    # NOVO: Crise de liquidez 2019 (repo market)
    "repo_crisis2019": dict(
        title="Crise do Repo Market 2019",
        type="B",
        tickers=[
            "SHY", "IEF", "TLT",   # treasuries
            "HYG", "LQD",           # crédito
            "XLF", "JPM", "BAC",    # bancos
            "SPY", "GLD",
        ],
        start="2018-01-01", end="2020-06-30",
        calm_search_to="2019-01-31",
        out=_out("v3_repo_crisis2019"),
    ),

    # ── TIPO C: EXÓGENA / MECÂNICA ────────────────────────────────────────────

    "covid2020": dict(
        title="COVID-19 2020 — Choque Exógeno",
        type="C",
        tickers=[
            "SPY", "QQQ", "IWM", "XLV", "XLP",
            "TLT", "GLD", "HYG", "EEM", "USO",
        ],
        start="2018-01-01", end="2021-12-31",
        calm_search_to="2019-12-31",
        out=_out("v3_covid2020"),
    ),

    "rates2022": dict(
        title="Choque de Juros Fed 2022",
        type="C",
        tickers=[
            "TLT", "IEF", "SHY",   # bonds
            "HYG", "LQD",           # crédito
            "SPY", "QQQ", "XLK",   # equities
            "GLD", "UUP",           # hedges
        ],
        start="2021-01-01", end="2023-12-31",
        calm_search_to="2021-12-31",
        out=_out("v3_rates2022"),
    ),

    "volmageddon2018": dict(
        title="Volmageddon 2018 — Colapso VIX",
        type="C",
        tickers=[
            "SPY", "QQQ", "IWM", "TLT", "GLD",
            "HYG", "XLF", "XLK", "XLV",
        ],
        start="2017-01-01", end="2019-12-31",
        calm_search_to="2018-01-31",
        out=_out("v3_volmageddon2018"),
    ),

    "flashcrash2010": dict(
        title="Flash Crash 2010 — Evento Mecânico",
        type="C",
        tickers=[
            "SPY", "QQQ", "IWM", "TLT", "GLD",
            "XLF", "XLK", "XLE", "XLV",
        ],
        start="2009-01-01", end="2012-06-30",
        calm_search_to="2010-01-31",
        out=_out("v3_flashcrash2010"),
    ),

    "brexit2016": dict(
        title="Brexit 2016 — Choque Político Exógeno",
        type="C",
        tickers=[
            "EWU", "EFA", "EWG",   # Europa
            "FXB", "FXE",           # moedas (GBP, EUR)
            "TLT", "GLD", "HYG",
            "SPY", "EEM",
        ],
        start="2015-01-01", end="2018-06-30",
        calm_search_to="2016-01-31",
        out=_out("v3_brexit2016"),
    ),

    # NOVO: Colapso do Silicon Valley Bank (SVB) 2023
    "svb2023": dict(
        title="SVB / Banking Crisis 2023 — Exógeno Regional",
        type="C",
        tickers=[
            "XLF", "JPM", "BAC", "C", "GS",   # grandes bancos
            "SPY", "TLT", "GLD",
            "HYG", "LQD",
        ],
        start="2022-01-01", end="2024-06-30",
        calm_search_to="2022-12-31",
        out=_out("v3_svb2023"),
    ),

    # ── NOVOS: ANÁLISE MULTI-MERCADO LONGA ────────────────────────────────────

    "global_2008_multiasset": dict(
        title="GFC 2008 — Visão Multi-Asset Global",
        type="A",
        tickers=ASSET_PRESETS["multi_asset"],
        start="2005-01-01", end="2011-12-31",
        calm_search_to="2007-01-31",
        out=_out("v3_gfc2008_multiasset"),
    ),

    "global_equity_2015": dict(
        title="2015–2016 — Sincronização Global de Equities",
        type="A",
        tickers=ASSET_PRESETS["global_equity"],
        start="2013-01-01", end="2017-12-31",
        calm_search_to="2014-12-31",
        out=_out("v3_global_equity_2015"),
    ),

    "commodities_2014": dict(
        title="Crash de Commodities 2014-2016",
        type="B",
        tickers=[
            "USO", "UNG", "GLD", "SLV",   # commodities
            "XLE", "XLB",                  # setores
            "EWZ", "EWC", "EWA",           # países dependentes
            "DBA",                          # agrícolas
        ],
        start="2012-01-01", end="2017-12-31",
        calm_search_to="2013-12-31",
        out=_out("v3_commodities_2014"),
    ),

    "em_crisis_2018": dict(
        title="Crise de EM 2018 — Turquia / Argentina",
        type="B",
        tickers=[
            "EEM", "EWZ", "EWY", "EWT", "EWW",   # EM
            "FXI",                                  # China
            "UUP",                                  # Dólar
            "GLD", "TLT", "SPY",
        ],
        start="2016-01-01", end="2020-06-30",
        calm_search_to="2017-12-31",
        out=_out("v3_em_crisis_2018"),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(name: str) -> None:
    """Executa um cenário pelo nome."""
    if name not in SCENARIOS:
        raise ValueError(f"Cenário '{name}' não encontrado. "
                         f"Disponíveis: {list(SCENARIOS.keys())}")
    meta = SCENARIOS[name]
    kw = {**V3_BASE, **{k: v for k, v in meta.items()
                        if k not in ("title", "type")}}
    cfg = ConfigV3(**kw)
    run(cfg, scenario_title=f"[{meta['type']}] {meta['title']}")


def run_all(type_filter: str = None) -> None:
    """
    Executa todos os cenários (ou filtrando por tipo A/B/C).
    """
    names = [n for n, m in SCENARIOS.items()
             if type_filter is None or m["type"] == type_filter]
    print(f"\nExecutando {len(names)} cenário(s)"
          + (f" [Tipo {type_filter}]" if type_filter else "") + "\n")
    for i, name in enumerate(names, 1):
        print(f"\n{'═'*72}")
        print(f"[{i}/{len(names)}] {name.upper()}: {SCENARIOS[name]['title']}")
        print(f"{'═'*72}")
        try:
            run_scenario(name)
        except Exception as e:
            print(f"  ❌ ERRO em {name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Kappa-FIN v3 — Cenários Expandidos")
    p.add_argument("scenario", nargs="?", default=None,
                   help=f"Nome do cenário ou 'all'. Disponíveis: {list(SCENARIOS.keys())}")
    p.add_argument("--type", default=None, choices=["A", "B", "C"],
                   help="Filtrar por tipo ao usar 'all'")
    p.add_argument("--list", action="store_true", help="Lista cenários disponíveis")
    a = p.parse_args()

    if a.list:
        print(f"\n{'CENÁRIO':<30} {'TIPO':<6} {'PERÍODO':<25} TÍTULO")
        print("─" * 90)
        for n, m in SCENARIOS.items():
            print(f"  {n:<28} [{m['type']}]   {m['start']}→{m['end']}   {m['title']}")
        print(f"\nTotal: {len(SCENARIOS)} cenários\n")
    elif a.scenario == "all" or a.scenario is None:
        run_all(type_filter=a.type)
    else:
        run_scenario(a.scenario)
