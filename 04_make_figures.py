"""
04_make_figures.py
Paper figures + a point-estimate summary table from results/summary.csv and results/alpha_*/*.npz

  python 04_make_figures.py --results results --alpha 0.1

Uses results/vif_het_alpha<a>.csv (heterogeneity-aware VIF, written by 06_extra_diagnostics.py; the
alpha tag uses "_" for ".", e.g. vif_het_alpha0_1.csv). If that file is missing, the VIF panel of Fig. 5
falls back to the homogeneous "vif" column of summary.csv and says so on the axis label.

table_main.csv/.tex are POINT-ESTIMATE summaries (coverage, width, interval score, homogeneous VIF and,
when available, heterogeneity-aware VIF) at 24/120/240 h. The paper's tables with 95% block-bootstrap
intervals are produced by 05_bootstrap_tables.py (results/table_main_ci*.csv), not by this script.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from conformal_lib import SEASON_NAMES

STYLE = {
    "gaussian": ("Gaussian (RMSE)", "tab:gray", "--"),
    "split_pooled": ("Split, pooled", "tab:blue", "-"),
    "split_local": ("Split, per-cell", "tab:orange", "-"),
    "mondrian_season": ("Per-cell + season", "tab:green", "-"),
    "aci_0.005": ("ACI per-cell (γ=0.005)", "tab:red", "-"),
    "aci_0.02": ("ACI per-cell (γ=0.02)", "tab:pink", "-"),
    "rolling_90": ("Rolling 90d", "tab:purple", "-"),
    "rolling_365": ("Rolling 365d", "tab:brown", "-"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--map_lead_h", type=int, default=120)
    a = ap.parse_args()
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(a.results, "summary.csv"))
    df = df[np.isclose(df.alpha, a.alpha)]
    het_path = os.path.join(a.results, "vif_het_alpha%s.csv" % str(a.alpha).replace(".", "_"))
    has_het = os.path.exists(het_path)
    if has_het:
        df = df.merge(pd.read_csv(het_path), on=["method", "lead_h"], how="left")
        vif_label = "Spatial overdispersion index (1 = independent)"
    else:
        print(f"WARNING: {het_path} not found (run 06_extra_diagnostics.py); "
              "Fig. 5 uses the homogeneous VIF from summary.csv instead")
        df = df.assign(vif_het=df.vif)
        vif_label = "Spatial overdispersion index, homogeneous ref. (1 = independent)"
    meta = np.load(os.path.join(a.results, "meta.npz"))
    leads_h, lat = meta["leads_h"], meta["lat"]
    methods = [m for m in STYLE if m in set(df.method)]
    R = {m: np.load(os.path.join(a.results, f"alpha_{a.alpha}", f"{m}.npz")) for m in methods}
    target = 1 - a.alpha
    li = int(np.argmin(np.abs(leads_h - a.map_lead_h)))

    def line(ax, key, ylabel, logy=False):
        for m in methods:
            lab, c, ls = STYLE[m]
            d = df[df.method == m].sort_values("lead_h")
            ax.plot(d.lead_h / 24, d[key], ls, color=c, marker="o", ms=3, label=lab)
            e = d[d.lead_h.isin([6, 12])]
            ax.plot(e.lead_h / 24, e[key], linestyle="none", marker="o", ms=5, mfc="white", mec=c, zorder=3)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    # Fig 1: coverage vs lead
    fig, ax = plt.subplots(figsize=(6, 4))
    line(ax, "coverage", "Test coverage (area-weighted)")
    ax.axhline(target, color="k", lw=1, ls=":")
    if np.isclose(a.alpha, 0.1):
        ax.set_ylim(0.85, 0.945)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig1_coverage_by_lead.png"), dpi=200); plt.close(fig)

    # Fig 2: width and interval score
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    line(axs[0], "width_K", "Mean interval width (K)")
    line(axs[1], "interval_score", "Interval score (K, lower is better)")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig2_width_score.png"), dpi=200); plt.close(fig)

    # Fig 3: conditional coverage by season and by year
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        axs[0].bar(np.arange(4) + i * wdt, R[m]["cov_season"][:, li], wdt, color=STYLE[m][1], label=STYLE[m][0])
        axs[1].plot(R[m]["years"], R[m]["cov_year"][:, li], STYLE[m][2], color=STYLE[m][1], marker="o", ms=3)
    axs[0].axhline(target, color="k", ls=":")
    axs[0].set_xticks(np.arange(4) + 0.4); axs[0].set_xticklabels(SEASON_NAMES)
    axs[0].set_ylim(max(0.5, target - 0.25), min(1.0, target + 0.1))
    axs[0].set_ylabel(f"Coverage at {leads_h[li]} h"); axs[0].set_title("By season of initialisation")
    axs[1].axhline(target, color="k", ls=":"); axs[1].set_title("By test year")
    if np.isclose(a.alpha, 0.1):
        axs[1].set_ylim(0.86, 0.95)
    axs[1].set_xlabel("Year"); axs[1].set_xticks(R[methods[0]]["years"])
    axs[0].legend(fontsize=6, loc="lower left", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_season_year.png"), dpi=200); plt.close(fig)

    # Fig 4: spatial coverage maps
    show = [m for m in ["split_pooled", "split_local", "aci_0.005"] if m in R]
    fig, axs = plt.subplots(1, len(show), figsize=(4.6 * len(show), 3.2), squeeze=False)
    order = np.argsort(lat)
    for ax, m in zip(axs[0], show):
        im = ax.imshow(R[m]["cov_map"][li][order], origin="lower", extent=[0, 360, lat.min(), lat.max()],
                       cmap="RdBu", vmin=target - 0.2, vmax=target + 0.2, aspect="auto")
        ax.set_title(STYLE[m][0], fontsize=9); ax.set_xlabel("Longitude")
    axs[0][0].set_ylabel("Latitude")
    fig.colorbar(im, ax=axs[0].tolist(), label=f"Coverage ({leads_h[li]} h lead)", shrink=0.9)
    fig.savefig(os.path.join(fig_dir, "fig4_coverage_maps.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # Fig 5: dependence diagnostics (overdispersion index, 5th percentile of daily spatial coverage,
    #        miss persistence minus shuffled-days null)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    line(axs[0], "vif_het", vif_label, logy=True)
    axs[0].axhline(1, color="k", ls=":")
    line(axs[1], "cov_day_q05", "5th percentile of daily spatial coverage")
    axs[1].axhline(target, color="k", ls=":")
    df = df.assign(persist_excess=df.miss_persist - df.miss_persist_null)
    line(axs[2], "persist_excess", "Miss persistence minus shuffled-days null")
    axs[2].axhline(0, color="k", ls=":")
    axs[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig5_dependence.png"), dpi=200); plt.close(fig)

    # Table: main numbers at three leads
    keep = df[df.lead_h.isin([24, 120, 240])].copy()
    tab = keep.pivot(index="method", columns="lead_h",
                     values=["coverage", "width_K", "interval_score", "vif"] + (["vif_het"] if has_het else [])).round(3)
    tab = tab.reindex([m for m in methods if m in tab.index])
    tab.to_csv(os.path.join(a.results, "table_main.csv"))
    with open(os.path.join(a.results, "table_main.tex"), "w") as f:
        f.write(tab.to_latex())
    print("figures ->", fig_dir)
    print(tab)


if __name__ == "__main__":
    main()
