"""Redraw Fig. B (AUC with 95% CI, internal vs external) from the numbers printed by the final Kaggle run
(rsna-multimodal-pneumonia-xai-final.ipynb, Sections 7 and 10). Same style as the notebook; fixes the overlapping
x-axis labels of the first version. Usage: python redraw_fig_auc_ci.py <output_folder>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
ORDER = ["tab_only_as", "tab_only_asv", "image", "fusion_as", "fusion_asv"]
COLOR = {"image": "#2a78d6", "fusion_as": "#eb6834", "fusion_asv": "#1baf7a", "tab_only_as": "#a3a29c", "tab_only_asv": "#52514e"}
NAME = {"image": "Image only (base)", "fusion_as": "Fusion: image + age + sex", "fusion_asv": "Fusion: image + age + sex + view",
        "tab_only_as": "Metadata: age + sex", "tab_only_asv": "Metadata: age + sex + view"}
# (AUC, CI low, CI high) copied from the notebook output (4 decimals; 0.5496 stands for the printed 0.5495,
# whose full-precision value rounds to 0.550 as in results_macros.tex)
RSNA = {"tab_only_as": (0.5826, 0.5354, 0.6266), "tab_only_asv": (0.7373, 0.7070, 0.7659), "image": (0.8869, 0.8712, 0.9022),
        "fusion_as": (0.8843, 0.8683, 0.9002), "fusion_asv": (0.8842, 0.8678, 0.9001)}
NIH_B = {"tab_only_as": (0.4709, 0.4526, 0.4897), "tab_only_asv": (0.5496, 0.5305, 0.5687), "image": (0.6749, 0.6578, 0.6912),
         "fusion_as": (0.6563, 0.6377, 0.6735), "fusion_asv": (0.6869, 0.6704, 0.7026)}

out = sys.argv[1] if len(sys.argv) > 1 else "."
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, (title, tab) in zip(axes, [("RSNA test (internal)", RSNA), ("NIH ChestX-ray14 (external, def. B)", NIH_B)]):
    yy = np.arange(len(ORDER))[::-1]
    for y, name in zip(yy, ORDER):
        a, lo, hi = tab[name]
        ax.errorbar(a, y, xerr=[[a - lo], [hi - a]], fmt="o", ms=5, color=COLOR[name], elinewidth=1.4, capsize=3)
        ax.annotate(f"{a:.3f}", (hi, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color="#0b0b0b")
    ax.set_yticks(yy, [NAME[n] for n in ORDER])
    ax.set_title(title)
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="y", visible=False)
fig.supxlabel("AUC (95% patient-level bootstrap CI)", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(out, "fig_auc_ci.png"), dpi=400, bbox_inches="tight")
fig.savefig(os.path.join(out, "fig_auc_ci.pdf"), bbox_inches="tight")
print("written to", out)
