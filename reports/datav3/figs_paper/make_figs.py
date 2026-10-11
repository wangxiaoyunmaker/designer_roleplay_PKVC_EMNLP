"""Generate 4 data-driven figures for the paper (ACL traditional style).

Outputs PDF (vector, for \\includegraphics) + PNG (preview) to ./figs/.

Inputs:
  /sessions/sleepy-compassionate-sagan/mnt/0512技术方案/data/processed/caps_quadruples.json
  /sessions/sleepy-compassionate-sagan/mnt/0512技术方案/reports/datav3/cross_model_full_metrics_v2.json
  /sessions/sleepy-compassionate-sagan/mnt/0512技术方案/data_v3/scale_responses.csv
  /sessions/sleepy-compassionate-sagan/mnt/0512技术方案/data_v3/task_responses.csv
  /sessions/sleepy-compassionate-sagan/mnt/0512技术方案/data/efa_loadings.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── ACL traditional style ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.color": "#dddddd",
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# cold tone palette (blue / teal / gray / muted red accent)
C_INHIBIT = "#3b6b96"    # cold blue
C_WEAK    = "#7a9cc6"    # light blue-gray
C_GUIDE   = "#d4843b"    # warm orange (positive moderate)
C_STRONG  = "#a33a3a"    # muted red (positive extreme)
C_GRAY    = "#6c6c6c"
C_NEUTRAL = "#aaaaaa"

OUT = Path("/sessions/sleepy-compassionate-sagan/mnt/outputs/figs")
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path("/sessions/sleepy-compassionate-sagan/mnt/0512技术方案")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    print(f"  saved: {name}.pdf + {name}.png")


# ════════════════════════════════════════════════════════════════════════
# Fig A: 36 quad effect-size × significance scatter
# ════════════════════════════════════════════════════════════════════════
def fig_A():
    print("Fig A: quad scatter ...")
    d = json.load(open(ROOT / "data/processed/caps_quadruples.json"))
    rows = []
    for q in d:
        rows.append({
            "abs_coef": abs(q["S_coef"]),
            "coef_sign": "+" if q["S_coef"] >= 0 else "-",
            "neg_log_p": -np.log10(max(q["S_p"], 1e-4)),
            "predicate": q["P"],
            "O": q["O"],
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    # 4 threshold regions (faint shading)
    # Strongly_forces: |c|>=0.50, p<0.05  -> x>=0.50, y>=-log10(0.05)=1.301
    # Guides:          0.20<=|c|<0.50, p<0.05 (positive only)
    # Weakly:          positive, 0.05<=p<0.10
    # Strictly_inhibits: negative, |c|>=0.20

    p05 = -np.log10(0.05)  # 1.301
    p10 = -np.log10(0.10)  # 1.000

    # background bands by p threshold
    ax.axhspan(p05, 5, color="#f0f4f9", alpha=0.55, zorder=0)
    ax.axhspan(p10, p05, color="#fafbfc", alpha=0.8, zorder=0)

    # threshold lines
    for x in (0.20, 0.50):
        ax.axvline(x, color="#cccccc", lw=0.6, ls="--", zorder=1)
    for y in (p10, p05):
        ax.axhline(y, color="#cccccc", lw=0.6, ls="--", zorder=1)

    style = {
        "strongly_forces_transition_to": dict(c=C_STRONG, marker="^", s=70, label=r"Strongly forces ($p{<}0.05$, $|\beta|{\geq}0.50$, +)"),
        "guides_transition_to":          dict(c=C_GUIDE,  marker="o", s=55, label=r"Guides ($p{<}0.05$, $0.20{\leq}|\beta|{<}0.50$, +)"),
        "weakly_shifts_to":              dict(c=C_WEAK,   marker="s", s=45, label=r"Weakly shifts ($0.05{\leq}p{<}0.10$, +)"),
        "strictly_inhibits":             dict(c=C_INHIBIT, marker="v", s=45, label=r"Strictly inhibits ($|\beta|{\geq}0.20$, $-$)"),
    }
    for pred, sty in style.items():
        sub = df[df["predicate"] == pred]
        ax.scatter(sub["abs_coef"], sub["neg_log_p"],
                   c=sty["c"], marker=sty["marker"], s=sty["s"],
                   edgecolors="white", linewidths=0.5,
                   label=sty["label"] + f"  (n={len(sub)})", zorder=3)

    # threshold annotations
    ax.text(0.51, 4.55, r"$|\beta|=0.50$", fontsize=7, color="#555", rotation=90, va="top")
    ax.text(0.21, 4.55, r"$|\beta|=0.20$", fontsize=7, color="#555", rotation=90, va="top")
    ax.text(0.92, p05 + 0.05, r"$p=0.05$", fontsize=7, color="#555", ha="right")
    ax.text(0.92, p10 + 0.05, r"$p=0.10$", fontsize=7, color="#555", ha="right")

    ax.set_xlabel(r"Standardized LASSO coefficient  $|\beta|$")
    ax.set_ylabel(r"Significance  $-\log_{10}(p)$")
    ax.set_xlim(0.18, 0.95)
    ax.set_ylim(0.9, 4.7)

    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fancybox=False, borderpad=0.5,
              handletextpad=0.4, labelspacing=0.4)

    n_total = len(df)
    n_pos = (df["coef_sign"] == "+").sum()
    n_neg = (df["coef_sign"] == "-").sum()
    ax.set_title(f"36 Ontology Quadruples by Predicate Category  "
                 f"($n_+={n_pos}$, $n_-={n_neg}$)", pad=6)

    plt.tight_layout()
    save(fig, "figA_quad_scatter")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# Fig B: cross-model module ladder line plot
# ════════════════════════════════════════════════════════════════════════
def fig_B():
    print("Fig B: cross-model ladder ...")
    d = json.load(open(ROOT / "reports/datav3/cross_model_full_metrics_v2.json"))

    # paper uses 5 models (excludes qwen27b)
    MODELS = [
        ("gpt4o",            "GPT-4o",           "o",  "#1f4e79"),
        ("gemini31flashlite","Gemini 3.1 FL",    "s",  "#2e7d5b"),
        ("claudehaiku45",    "Claude Haiku 4.5", "^",  "#a14d2a"),
        ("deepseekv4flash",  "DeepSeek V4 Flash","D",  "#6c3a8e"),
        ("qwen35b",          "Qwen 3.6 35B",     "v",  "#9c1a3e"),
    ]
    LADDER = ["b0", "b1", "b2", "b3", "v2"]
    XTICKS = ["B0", "B1", "B2", "B3", r"B4$^\star$"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Left: chain diversity (5/5 monotonic)
    ax = axes[0]
    for slug, name, mk, color in MODELS:
        vals = [d[slug]["metrics"][c]["seg2_chain_diversity"] for c in LADDER]
        ax.plot(range(5), vals, marker=mk, color=color, label=name,
                lw=1.2, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(XTICKS)
    ax.set_ylabel("Chain diversity " + r"($\uparrow$)")
    ax.set_xlabel("Module ladder configuration")
    ax.set_title("(a) Chain diversity: 5/5 monotonic", fontsize=9, pad=4)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_ylim(0.55, 0.97)

    # Right: macro-TED (lower better)
    ax = axes[1]
    for slug, name, mk, color in MODELS:
        vals = [d[slug]["metrics"][c]["seg2_macro_TED"] for c in LADDER]
        ax.plot(range(5), vals, marker=mk, color=color, label=name,
                lw=1.2, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(XTICKS)
    ax.set_ylabel("macro-TED " + r"($\downarrow$)")
    ax.set_xlabel("Module ladder configuration")
    ax.set_title("(b) Chain accuracy: 5/5 net improvement", fontsize=9, pad=4)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # shared legend on right
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               handletextpad=0.4, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "figB_module_ladder")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# Fig C: CAPS ablation Trait-pred ρ × 5 model bar
# ════════════════════════════════════════════════════════════════════════
def fig_C():
    print("Fig C: CAPS ablation bar ...")
    d = json.load(open(ROOT / "reports/datav3/cross_model_full_metrics_v2.json"))

    MODELS = [
        ("gpt4o",            "GPT-4o"),
        ("gemini31flashlite","Gemini 3.1\nFL"),
        ("claudehaiku45",    "Claude\nHaiku 4.5"),
        ("deepseekv4flash",  "DeepSeek\nV4 Flash"),
        ("qwen35b",          "Qwen 3.6\n35B"),
    ]
    metric_key = "seg1_trait_pred_rho"

    v2_vals       = [d[s]["metrics"]["v2"][metric_key]         for s, _ in MODELS]
    no_caps_vals  = [d[s]["metrics"]["v2_no_caps"][metric_key] for s, _ in MODELS]
    deltas        = [a - b for a, b in zip(v2_vals, no_caps_vals)]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(MODELS))
    w = 0.34

    bars1 = ax.bar(x - w/2, v2_vals,      w, label=r"$\mathbf{B4}$ (with CAPS)",
                   color=C_INHIBIT, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, no_caps_vals, w, label=r"$\mathbf{B4}$-NoCaps",
                   color=C_NEUTRAL, edgecolor="white", linewidth=0.5)

    # zero line
    ax.axhline(0, color="#444444", lw=0.7, zorder=2)

    # Δ annotation above each pair
    for i, dlt in enumerate(deltas):
        top = max(v2_vals[i], no_caps_vals[i])
        ax.annotate(f"Δ = {dlt:+.03f}",
                    xy=(i, top + 0.012),
                    ha="center", va="bottom", fontsize=7,
                    color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in MODELS])
    ax.set_ylabel(r"Trait-prediction Spearman $\rho$ (seg1) $\uparrow$")
    ax.set_title("Removing CAPS Routing Block: 5/5 models show $\\rho$ decrease",
                 pad=6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fancybox=False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_ylim(-0.06, max(v2_vals) + 0.07)

    plt.tight_layout()
    save(fig, "figC_caps_ablation_rho")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# Fig D: trait-continuous manifold (4-d EFA factor → t-SNE)
# ════════════════════════════════════════════════════════════════════════
def fig_D():
    print("Fig D: trait manifold ...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Load scale + loadings → 4-d factor scores
    scales = pd.read_csv(ROOT / "data_v3/scale_responses.csv")
    loadings = pd.read_csv(ROOT / "data/efa_loadings.csv").set_index("question_node")

    M = scales.pivot_table(index="user_id", columns="question_node",
                            values="score", aggfunc="mean")
    if M.isna().any().any():
        M = M.fillna(M.mean(numeric_only=True))
    common = [q for q in loadings.index if q in M.columns]
    L = loadings.loc[common].values
    M_std = (M[common].values - M[common].values.mean(axis=0)) / (M[common].values.std(axis=0) + 1e-8)
    F = M_std @ L  # N x 4

    # Load Q1 (first reaction) dominant PKVC per designer for color
    tasks = pd.read_csv(ROOT / "data_v3/task_responses.csv")
    # Q1 option labels → PKVC factor mapping (from L4c)
    Q1_OPT_TO_PKVC = {"A": "C", "B": "P", "C": "K", "D": "V"}
    q1 = tasks[tasks["question_id"] == "Q1"]
    if len(q1) == 0:
        # try alternative qid column
        q1 = tasks[tasks.get("qid", "") == "Q1"]
    print(f"  Q1 rows: {len(q1)}; tasks columns: {tasks.columns.tolist()[:8]}")

    # answer_value 是 '"A. ..."' 整段, 抽首字母
    import re
    def extract_letter(s):
        m = re.match(r'^["\s]*([ABCD])\b', str(s))
        return m.group(1) if m else None

    q1 = tasks[tasks["question_id"] == "Q1"][["user_id", "answer_value"]].copy()
    q1["letter"] = q1["answer_value"].apply(extract_letter)
    q1["pkvc"] = q1["letter"].map(Q1_OPT_TO_PKVC)
    print(f"  Q1 letter dist: {q1['letter'].value_counts().to_dict()}")
    pkvc_map = dict(zip(q1["user_id"].astype(str), q1["pkvc"]))

    user_ids = M.index.astype(str).tolist()
    colors_by_pkvc = {"P": "#a33a3a", "K": "#3b6b96", "V": "#2e7d5b", "C": "#d4843b"}
    point_pkvc = [pkvc_map.get(u, "?") for u in user_ids]
    point_color = [colors_by_pkvc.get(p, "#cccccc") for p in point_pkvc]

    # t-SNE on 4-d factor scores
    F_std = StandardScaler().fit_transform(F)
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=42, learning_rate="auto")
    Z = tsne.fit_transform(F_std)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for pkvc, color in colors_by_pkvc.items():
        mask = [p == pkvc for p in point_pkvc]
        n = sum(mask)
        if n == 0: continue
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=color, s=22, edgecolors="white", linewidths=0.4,
                   label=f"{pkvc}  (n={n})", alpha=0.85, zorder=3)

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("4-d EFA factor manifold of 163 designers\ncolored by Q1 first-reaction PKVC dominant",
                 fontsize=9, pad=6)
    ax.legend(title="Dominant PKVC", loc="best", frameon=True,
              framealpha=0.95, edgecolor="#cccccc", fancybox=False,
              fontsize=7, title_fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    save(fig, "figD_trait_manifold_tsne")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_A()
    fig_B()
    fig_C()
    fig_D()
    print("\nAll figures saved to:", OUT)
