# ============================================================
# SCRIPT 10 — Karar Eğrisi Analizi (Decision Curve Analysis, DCA)
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumu: R1-71
#   "Karar eğrisi analizi modelin klinik uygunluğunu güçlendirir."
#
# NE YAPAR:
#   Vickers & Elkin (2006) net fayda (net benefit) çerçevesini çok-sınıf
#   probleme one-vs-rest olarak uygular. Her azınlık sınıf için ayrı eğri.
#
#   Net Benefit(p_t) = TP/N − (FP/N) × (p_t / (1 − p_t))
#     p_t : eşik olasılık (klinisyenin "bu olasılığın üstünde müdahale ederim"
#           dediği nokta; tedavi/tetkik maliyeti ile kaçırma maliyetinin oranı)
#
#   Karşılaştırma stratejileri:
#     - Model (HFS+SMOTE pipeline, olasılık çıktısı)
#     - Treat-all   : herkesi hasta say
#     - Treat-none  : kimseyi hasta sayma  (net fayda = 0)
#     - Klinik kural: TSH/TT4 eşik kuralı (sabit karar; R1-59 ile bağlantılı)
#
#   ÖNEMLİ: Eşik aralığı azınlık sınıf prevalansına göre seçilir. Prevalans
#   %2.3 iken p_t=0.5 klinik olarak anlamsızdır; bu nedenle p_t ∈ [0.01, 0.30].
#
# ÇIKTI:
#   dca_net_benefit.csv        (tüm eğri noktaları)
#   dca_summary.csv            (özet: hangi aralıkta model üstün)
#   dca_report.txt
#   figures/dca_{class}.png
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_dca_results"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

F_NB      = os.path.join(OUTPUT_DIR, "dca_net_benefit.csv")
F_SUMMARY = os.path.join(OUTPUT_DIR, "dca_summary.csv")
F_REPORT  = os.path.join(OUTPUT_DIR, "dca_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'
CLASS_NAMES = {0: "overt_hypothyroid", 1: "subclinical_hypothyroid", 2: "euthyroid"}
CLASS_DISPLAY = {0: "Overt hypothyroid", 1: "Subclinical hypothyroid", 2: "Euthyroid"}

# Azınlık sınıflar için DCA (çoğunluk sınıfın DCA'sı klinik olarak anlamsız)
DCA_CLASSES = [0, 1]

# Eşik olasılık aralığı — düşük prevalansa uygun
THRESHOLDS = np.arange(0.01, 0.31, 0.01)

# Klinik kural eşikleri (R1-59 scripti ile aynı)
DENORM = {"TSH": 1000.0, "T3": 100.0, "TT4": 1000.0, "T4U": 10.0, "FTI": 1000.0, "age": 100.0}
TSH_HIGH, TT4_LOW = 4.0, 60.0


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
    }

PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}


# ============================================================
# HFS
# ============================================================

def information_gain(X_col, y):
    return mutual_info_classif(X_col.reshape(-1, 1), y,
                               discrete_features='auto', random_state=INTERNAL_SEED)[0]

def gain_ratio(X_col, y):
    IG = information_gain(X_col, y)
    n_unique = len(np.unique(X_col))
    if n_unique <= 10:
        _, counts = np.unique(X_col, return_counts=True)
    else:
        n_bins = max(10, int(np.sqrt(len(X_col))))
        counts = np.histogram(X_col, bins=n_bins)[0]; counts = counts[counts > 0]
    probs = counts / counts.sum()
    Hx = -np.sum(probs * np.log2(probs))
    return IG / (Hx + 1e-12) if Hx > 0 else 0.0

def hfs_select(Xtr, ytr, k=K_SELECT, theta=0.90, w=0.5):
    feats = list(Xtr.columns); X = Xtr.values
    redundant = set(); n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                redundant.add(feats[j])
    non_red = [f for f in feats if f not in redundant]
    scores = []
    for f in non_red:
        xf = Xtr[f].values
        scores.append((f, w * information_gain(xf, ytr) + (1 - w) * gain_ratio(xf, ytr)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores[:min(k, len(scores))]]


# ============================================================
# Net fayda (Vickers & Elkin)
# ============================================================

def net_benefit(y_true_bin, y_prob, threshold):
    """
    Net Benefit = TP/N − (FP/N) × (p_t / (1 − p_t))
    y_true_bin : 0/1 (bu sınıf mı?)
    y_prob     : bu sınıfın tahmin olasılığı
    """
    n = len(y_true_bin)
    if n == 0:
        return np.nan
    pred_pos = y_prob >= threshold
    tp = np.sum(pred_pos & (y_true_bin == 1))
    fp = np.sum(pred_pos & (y_true_bin == 0))
    w = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * w


def net_benefit_treat_all(y_true_bin, threshold):
    """Herkesi pozitif say."""
    n = len(y_true_bin)
    tp = np.sum(y_true_bin == 1)
    fp = np.sum(y_true_bin == 0)
    w = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * w


def net_benefit_fixed_decision(y_true_bin, pred_bin, threshold):
    """Sabit karar veren strateji (klinik kural gibi) için net fayda."""
    n = len(y_true_bin)
    tp = np.sum((pred_bin == 1) & (y_true_bin == 1))
    fp = np.sum((pred_bin == 1) & (y_true_bin == 0))
    w = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * w


def clinical_rule_predict_raw(df_norm):
    """Normalize veriden ham ölçeğe çevirip klinik kuralı uygular."""
    tsh = df_norm["TSH"].values * DENORM["TSH"]
    tt4 = df_norm["TT4"].values * DENORM["TT4"]
    pred = np.full(len(df_norm), 2, dtype=int)
    hi = tsh > TSH_HIGH
    pred[hi & (tt4 < TT4_LOW)] = 0
    pred[hi & (tt4 >= TT4_LOW)] = 1
    return pred


# ============================================================
# Ana döngü
# ============================================================

def run_all():
    print("=" * 64)
    print("KARAR EĞRİSİ ANALİZİ (DCA) — ann-thyroid")
    print("=" * 64)
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek")
    print(f"Eşik aralığı: p_t ∈ [{THRESHOLDS[0]:.2f}, {THRESHOLDS[-1]:.2f}]")
    print(f"DCA sınıfları: {[CLASS_DISPLAY[c] for c in DCA_CLASSES]}")
    print("=" * 64)

    nb_rows = []

    for run_id in SPLIT_SEEDS:
        X = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=run_id, stratify=y)
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)
        feats = hfs_select(Xtr, ytr.values)
        XtrH, XteH = Xtr[feats], Xte[feats]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

        # klinik kural tahminleri (ham ölçekten)
        rule_pred = clinical_rule_predict_raw(Xte_raw)

        for mname, model in get_models().items():
            t0 = time.time()
            try:
                pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)),
                                    ("model", model)])
                grid = GridSearchCV(pipe, PARAM_GRIDS[mname], cv=cv,
                                    scoring="roc_auc_ovr_weighted", n_jobs=-1)
                grid.fit(XtrH, ytr)
                prob = grid.best_estimator_.predict_proba(XteH)
            except Exception as e:
                print(f"  ✗ {mname} seed={run_id} HATA: {e}")
                continue

            for c in DCA_CLASSES:
                y_bin = (yte.values == c).astype(int)
                p_c = prob[:, c]
                rule_bin = (rule_pred == c).astype(int)
                prevalence = y_bin.mean()

                for t in THRESHOLDS:
                    nb_rows.append({
                        "run_id": run_id, "model": mname,
                        "class": CLASS_NAMES[c], "threshold": round(float(t), 3),
                        "nb_model": net_benefit(y_bin, p_c, t),
                        "nb_treat_all": net_benefit_treat_all(y_bin, t),
                        "nb_treat_none": 0.0,
                        "nb_clinical_rule": net_benefit_fixed_decision(y_bin, rule_bin, t),
                        "prevalence": round(float(prevalence), 4),
                    })
            print(f"  seed={run_id} {mname:<20} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(nb_rows)
    df.to_csv(F_NB, index=False)
    print(f"\n  ✓ {F_NB}")
    aggregate(df)
    make_figures(df)


def aggregate(df):
    # eşik × sınıf × model ortalaması
    g = df.groupby(["model", "class", "threshold"]).agg(
        nb_model=("nb_model", "mean"),
        nb_treat_all=("nb_treat_all", "mean"),
        nb_clinical_rule=("nb_clinical_rule", "mean"),
        prevalence=("prevalence", "mean"),
    ).round(5).reset_index()

    # model hangi aralıkta en iyi?
    g["best_strategy"] = g.apply(
        lambda r: max(
            [("model", r["nb_model"]), ("treat_all", r["nb_treat_all"]),
             ("treat_none", 0.0), ("clinical_rule", r["nb_clinical_rule"])],
            key=lambda x: x[1])[0], axis=1)
    g.to_csv(F_SUMMARY, index=False)
    print(f"  ✓ {F_SUMMARY}")

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\nKARAR EĞRİSİ ANALİZİ (DCA) — ann-thyroid\n")
        f.write("R1-71 · Vickers & Elkin net fayda çerçevesi (one-vs-rest)\n")
        f.write("=" * 72 + "\n\n")
        f.write("Net Benefit = TP/N − (FP/N) × (p_t / (1 − p_t))\n")
        f.write("p_t: klinisyenin müdahale eşiği (tetkik/tedavi maliyeti oranı)\n\n")

        for cls in g["class"].unique():
            sub = g[g["class"] == cls]
            prev = sub["prevalence"].iloc[0]
            f.write(f"\n[{cls}]  prevalans = {prev*100:.1f}%\n")
            f.write("-" * 72 + "\n")
            for mname in sub["model"].unique():
                s = sub[sub.model == mname]
                # modelin treat-all ve treat-none'ı geçtiği aralık
                better = s[(s.nb_model > s.nb_treat_all) & (s.nb_model > 0)]
                if len(better):
                    lo, hi = better["threshold"].min(), better["threshold"].max()
                    f.write(f"  {mname:<20} model üstün: p_t ∈ [{lo:.2f}, {hi:.2f}]  "
                            f"({len(better)}/{len(s)} eşik)\n")
                else:
                    f.write(f"  {mname:<20} model hiçbir eşikte treat-all'ı geçmiyor\n")
                # kurala göre
                vs_rule = s[s.nb_model > s.nb_clinical_rule]
                f.write(f"  {'':<20} klinik kurala üstün: {len(vs_rule)}/{len(s)} eşik\n")

            # örnek eşiklerde tablo
            f.write(f"\n  Örnek eşiklerde net fayda:\n")
            f.write(f"  {'p_t':<8}{'Model(en iyi)':<18}{'Treat-all':<14}{'Klinik kural':<14}\n")
            for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
                s = sub[np.isclose(sub.threshold, t)]
                if len(s):
                    best_m = s.loc[s["nb_model"].idxmax()]
                    f.write(f"  {t:<8.2f}{best_m['nb_model']:<18.5f}"
                            f"{best_m['nb_treat_all']:<14.5f}"
                            f"{best_m['nb_clinical_rule']:<14.5f}\n")

        f.write("\n\nYORUM:\n" + "-" * 72 + "\n")
        f.write("  Net fayda > treat-all ve > 0 olan eşik aralığında model klinik\n")
        f.write("  olarak kullanışlıdır. Düşük prevalanslı sınıflarda treat-all\n")
        f.write("  stratejisi hızla negatife döner; bu nedenle modelin dar bir eşik\n")
        f.write("  aralığında bile üstün olması klinik değer taşır.\n")
    print(f"  ✓ {F_REPORT}")

    print("\nÖZET:")
    for cls in g["class"].unique():
        sub = g[g["class"] == cls]
        for mname in sub["model"].unique():
            s = sub[sub.model == mname]
            better = s[(s.nb_model > s.nb_treat_all) & (s.nb_model > 0)]
            print(f"  {cls:<26} {mname:<20} model üstün: {len(better)}/{len(s)} eşik")


def make_figures(df):
    for c_name in df["class"].unique():
        sub = df[df["class"] == c_name]
        g = sub.groupby(["model", "threshold"]).agg(
            nb_model=("nb_model", "mean"),
            nb_treat_all=("nb_treat_all", "mean"),
            nb_clinical_rule=("nb_clinical_rule", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(7.5, 5))
        colors = {"Logistic Regression": "#4c72b0", "Random Forest": "#c44e52",
                  "XGBoost": "#55a868"}
        for mname in g["model"].unique():
            s = g[g.model == mname]
            ax.plot(s["threshold"], s["nb_model"], "-", lw=2.2,
                    color=colors.get(mname, "gray"), label=f"{mname} (HFS+SMOTE)", zorder=3)
        # referans stratejiler
        s0 = g[g.model == g["model"].iloc[0]]
        ax.plot(s0["threshold"], s0["nb_treat_all"], "--", color="#888888", lw=1.6,
                label="Treat all", zorder=2)
        ax.axhline(0, color="black", ls=":", lw=1.2, label="Treat none", zorder=2)
        ax.plot(s0["threshold"], s0["nb_clinical_rule"], "-.", color="#dd8452", lw=1.8,
                label="Clinical rule (TSH/TT4)", zorder=2)

        ax.set_xlabel("Threshold probability $p_t$", fontsize=11)
        ax.set_ylabel("Net benefit", fontsize=11)
        ax.set_title(f"Decision Curve Analysis — {c_name.replace('_', ' ')}",
                     fontsize=12.5, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"dca_{c_name}.png"),
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✓ figures/dca_{c_name}.png")


if __name__ == "__main__":
    run_all()
    print("\n" + "=" * 64 + "\nDONE\n" + "=" * 64)
