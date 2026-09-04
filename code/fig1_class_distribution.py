# ============================================================
# Figure 1 — Class Distribution (ann-thyroid, 7,200 records)
#
# Sınıf adlandırması hormon profiline göre doğrulanmıştır:
#   Class 0 (n=166) : TSH medyan 53.0 mIU/L (↑↑), TT4 31.6 nmol/L (↓)
#                     → overt hypothyroid  (aşikar hipotiroidi)
#   Class 1 (n=368) : TSH medyan 9.7 mIU/L (↑), TT4 90.0 (normal-alt)
#                     → subclinical hypothyroid (subklinik hipotiroidi)
#   Class 2 (n=6666): TSH medyan 1.5, TT4 109.0 → euthyroid (ötiroid)
#
# NOT: ann-thyroid'in bazı dağıtımlarında Class 0 "hyperthyroid" olarak
# etiketlenir; bu biyokimyasal profille tutarsızdır (bu sınıfta TSH < 0.4
# olan hiç kayıt yoktur). Bkz. makale Section 2.1.
#
# Çıktı: Figure1_class_distribution.png (300 DPI)
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_PATH   = "thyroid_ann_7200.csv"      # <- Veri seti yolunu güncelle
OUTPUT_PATH = "Figure1_class_distribution.png"
TARGET_COL  = 'target'

# Doğrulanmış sınıf adları (kod 1,2,3 → 0,1,2)
CLASS_NAMES = {
    0: "Overt\nhypothyroid",
    1: "Subclinical\nhypothyroid",
    2: "Euthyroid",
}
# Azınlık sınıfları vurgulamak için renk
CLASS_COLORS = ["#c44e52", "#dd8452", "#4c72b0"]   # kırmızı, turuncu, mavi

# ============================================================
# VERİ
# ============================================================
print("=" * 60)
print("Figure 1 — Class Distribution")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
y = data[TARGET_COL] - 1                   # 1,2,3 → 0,1,2

counts = y.value_counts().sort_index()
total = counts.sum()
pcts = (counts / total * 100).round(1)

print(f"Toplam: {total} kayıt")
for c in sorted(counts.index):
    label = CLASS_NAMES[c].replace("\n", " ")
    print(f"  Class {c} ({label:<26}): n={counts[c]:>5}  ({pcts[c]:>4.1f}%)")

imbalance_ratio = counts.max() / counts.min()
print(f"\nDengesizlik oranı (max/min): {imbalance_ratio:.1f}:1")

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5))

x = np.arange(len(counts))
bars = ax.bar(
    x, counts.values,
    color=CLASS_COLORS,
    edgecolor="black", linewidth=0.8,
    width=0.62, zorder=3,
)

# Bar üstü etiketler: n + yüzde
for i, (bar, n, p) in enumerate(zip(bars, counts.values, pcts.values)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.012,
        f"n = {n:,}\n({p}%)",
        ha="center", va="bottom",
        fontsize=10.5, fontweight="bold",
        zorder=4,
    )

ax.set_xticks(x)
ax.set_xticklabels([CLASS_NAMES[c] for c in sorted(counts.index)], fontsize=10.5)
ax.set_ylabel("Number of instances", fontsize=11)
ax.set_xlabel("")
ax.set_ylim(0, counts.max() * 1.18)

ax.set_title(
    "Class Distribution of the ann-thyroid Dataset\n"
    f"(N = {total:,};  imbalance ratio {imbalance_ratio:.1f}:1)",
    fontsize=12.5, fontweight="bold", pad=12,
)

ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nFigure saved -> {OUTPUT_PATH}")
