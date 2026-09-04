# ============================================================
# SCRIPT 9 — Sınıf-Bazlı SHAP + Sıralama Uyumu
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1-72, R1-73, R1-80, R1-81
#   #72: "Hyperthyroid, hypothyroid ve normal sınıfları için AYRI SHAP
#         açıklamaları verilmeli."
#   #73: "SHAP değerlerini sınıflar arasında ortalamak, hastalığa özgü
#         yorumlama örüntülerini gizleyebilir."
#   #80: "HFS sıralaması, SHAP sıralaması ve permutation importance sıralaması
#         nicel olarak karşılaştırılmalı (örn. Spearman korelasyonu)."
#   #81: "Doğru ve yanlış sınıflandırılmış temsili azınlık vakalar için yerel
#         SHAP açıklamaları eklenmeli."
#
# NE YAPAR:
#   1) Her sınıf için AYRI SHAP önem sıralaması (ortalama YOK)  → #72, #73
#   2) HFS / SHAP / Permutation sıralamalarının Spearman uyumu   → #80
#   3) Sınıf-bazlı SHAP summary plot (3 sınıf × model)           → #72
#   4) Temsili doğru/yanlış sınıflanmış azınlık vakalar için
#      yerel SHAP katkı tablosu                                  → #81
#
#   Tüm analiz 10 split üzerinde tekrarlanır (#76/#77/#78 ile tutarlı olsun diye),
#   sıralama uyumu split'ler arası ortalama ± std olarak raporlanır.
#
# ÇIKTI:
#   shap_per_class_importance.csv   (#72, #73)
#   ranking_agreement.csv           (#80)
#   local_shap_cases.csv            (#81)
#   shap_per_class_report.txt
#   figures/shap_perclass_{model}.png  (#72)
#
# GEREKLİ: shap  (pip install shap)
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

try:
    import shap
except ImportError:
    raise SystemExit("shap gerekli: pip install shap")

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
SHAP_SAMPLE = 300
FIG_SEED = 5          # figürler median-performance run'dan (makale ile tutarlı)

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_shap_perclass_results"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

F_PERCLASS = os.path.join(OUTPUT_DIR, "shap_per_class_importance.csv")
F_AGREE    = os.path.join(OUTPUT_DIR, "ranking_agreement.csv")
F_LOCAL    = os.path.join(OUTPUT_DIR, "local_shap_cases.csv")
F_REPORT   = os.path.join(OUTPUT_DIR, "shap_per_class_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

# Sınıf adları — hormon profiline göre doğrulanmış (bkz. makale §2.1)
CLASS_NAMES = {0: "overt_hypothyroid", 1: "subclinical_hypothyroid", 2: "euthyroid"}
CLASS_DISPLAY = {0: "Overt hypothyroid", 1: "Subclinical hypothyroid", 2: "Euthyroid"}

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
    }

PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Decision Tree":       {"model__max_depth": [5, 10, None]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}
SCORING_CV = 'roc_auc_ovr_weighted'


# ============================================================
# HFS — skorları da döndürür (#80 için sıralama gerekli)
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

def hfs_select_with_scores(Xtr, ytr, k=K_SELECT, theta=0.90, w=0.5):
    """Seçilen özellikleri VE tüm HFS skorlarını döndürür (#80 için)."""
    feats = list(Xtr.columns); X = Xtr.values
    redundant = set(); n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                redundant.add(feats[j])
    non_red = [f for f in feats if f not in redundant]
    scores = {}
    for f in non_red:
        xf = Xtr[f].values
        scores[f] = w * information_gain(xf, ytr) + (1 - w) * gain_ratio(xf, ytr)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ordered[:min(k, len(ordered))]]
    return selected, scores


# ============================================================
# Sınıf-bazlı SHAP (#72, #73) — ORTALAMA YOK
# ============================================================

def shap_values_per_class(model, X_background, X_explain, feat_names, model_name):
    """
    Dönüş: {class_idx: (n_samples × n_features) SHAP matrisi}
    Sınıflar ORTALANMAZ — her sınıf ayrı tutulur (#73'ün talebi).
    """
    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_background)
        sv = explainer.shap_values(X_explain)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain)

    out = {}
    nf = len(feat_names)

    if isinstance(sv, list):
        # list of (n, feat) — her eleman bir sınıf
        for c, arr in enumerate(sv):
            out[c] = np.asarray(arr)
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            if sv.shape[-1] == nf:          # (class, n, feat)
                for c in range(sv.shape[0]):
                    out[c] = sv[c]
            else:                            # (n, feat, class)
                for c in range(sv.shape[2]):
                    out[c] = sv[:, :, c]
        elif sv.ndim == 2:
            # binary/tek çıktı — tek sınıf gibi ele al
            out[0] = sv
    return out


# ============================================================
# Ana döngü
# ============================================================

def run_all():
    print("=" * 64)
    print("SINIF-BAZLI SHAP + SIRALAMA UYUMU — ann-thyroid")
    print("=" * 64)
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek | SHAP örneklem: {SHAP_SAMPLE}")
    print(f"Modeller: {list(get_models().keys())}")
    print(f"Sınıflar AYRI analiz edilir (ortalama YOK) — R1-73")
    print("=" * 64)

    perclass_rows, agree_rows, local_rows = [], [], []

    for run_id in SPLIT_SEEDS:
        X = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=run_id, stratify=y)
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)

        feats, hfs_scores = hfs_select_with_scores(Xtr, ytr.values)
        XtrH, XteH = Xtr[feats], Xte[feats]

        # SHAP açıklama örneklemi — EĞİTİMDE KULLANILMAYAN test verisinden (#75)
        n_ex = min(SHAP_SAMPLE, len(XteH))
        rng = np.random.RandomState(INTERNAL_SEED + run_id)
        ex_idx = rng.choice(len(XteH), n_ex, replace=False)
        X_explain = XteH.iloc[ex_idx]
        y_explain = yte.iloc[ex_idx].values
        X_bg = XtrH.sample(min(100, len(XtrH)), random_state=INTERNAL_SEED)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)

        for mname, model in get_models().items():
            t0 = time.time()
            try:
                pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)),
                                    ("model", model)])
                grid = GridSearchCV(pipe, PARAM_GRIDS[mname], cv=cv,
                                    scoring=SCORING_CV, n_jobs=-1)
                grid.fit(XtrH, ytr)
                fitted = grid.best_estimator_.named_steps["model"]

                # --- (#72/#73) sınıf-bazlı SHAP
                sv_by_class = shap_values_per_class(fitted, X_bg, X_explain, feats, mname)

                for c, mat in sv_by_class.items():
                    if c not in CLASS_NAMES:
                        continue
                    imp = np.abs(mat).mean(axis=0)          # bu SINIF için önem
                    for f, v in zip(feats, imp):
                        perclass_rows.append({
                            "run_id": run_id, "model": mname,
                            "class": CLASS_NAMES[c], "feature": f,
                            "mean_abs_shap": round(float(v), 6),
                        })

                # --- (#80) permutation importance
                perm = permutation_importance(
                    grid.best_estimator_, XteH, yte,
                    n_repeats=10, random_state=INTERNAL_SEED, n_jobs=-1,
                    scoring="f1_macro")   # REV: was "accuracy" (R2 c.4)
                perm_imp = dict(zip(feats, perm.importances_mean))

                # --- (#80) SHAP genel önem (sınıflar arası ort — sadece sıralama için)
                shap_overall = {}
                for f_i, f in enumerate(feats):
                    vals = [np.abs(mat[:, f_i]).mean() for c, mat in sv_by_class.items()
                            if c in CLASS_NAMES]
                    shap_overall[f] = float(np.mean(vals)) if vals else 0.0

                # --- (#80) üç sıralamayı Spearman ile karşılaştır
                hfs_v = [hfs_scores.get(f, 0.0) for f in feats]
                shap_v = [shap_overall[f] for f in feats]
                perm_v = [perm_imp[f] for f in feats]

                for a_name, a_v, b_name, b_v in [
                    ("HFS", hfs_v, "SHAP", shap_v),
                    ("HFS", hfs_v, "Permutation", perm_v),
                    ("SHAP", shap_v, "Permutation", perm_v),
                ]:
                    rho, p = spearmanr(a_v, b_v)
                    agree_rows.append({
                        "run_id": run_id, "model": mname,
                        "ranking_a": a_name, "ranking_b": b_name,
                        "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else np.nan,
                        "p_value": round(float(p), 5) if not np.isnan(p) else np.nan,
                        "n_features": len(feats),
                    })

                # --- (#81) yerel SHAP: doğru/yanlış sınıflanmış azınlık vakalar
                if run_id == FIG_SEED:
                    y_pred_ex = grid.best_estimator_.predict(X_explain)
                    for c in [0, 1]:      # azınlık sınıflar
                        if c not in sv_by_class:
                            continue
                        mask_c = (y_explain == c)
                        if mask_c.sum() == 0:
                            continue
                        correct = mask_c & (y_pred_ex == c)
                        wrong = mask_c & (y_pred_ex != c)
                        for label, m in [("correct", correct), ("misclassified", wrong)]:
                            idxs = np.where(m)[0]
                            if len(idxs) == 0:
                                continue
                            i = int(idxs[0])         # temsili ilk vaka
                            contrib = sv_by_class[c][i]
                            for f, v, xval in zip(feats, contrib, X_explain.iloc[i].values):
                                local_rows.append({
                                    "run_id": run_id, "model": mname,
                                    "true_class": CLASS_NAMES[c],
                                    "pred_class": CLASS_NAMES.get(int(y_pred_ex[i]), "?"),
                                    "case_type": label, "case_index": i,
                                    "feature": f,
                                    "feature_value_scaled": round(float(xval), 4),
                                    "shap_contribution": round(float(v), 6),
                                })

                    # --- (#72) sınıf-bazlı figür
                    make_perclass_figure(sv_by_class, X_explain, feats, mname)

                top_str = {}
                for c, mat in sv_by_class.items():
                    if c in CLASS_NAMES:
                        imp = np.abs(mat).mean(axis=0)
                        top_str[CLASS_DISPLAY[c][:12]] = feats[int(np.argmax(imp))]
                print(f"  seed={run_id} {mname:<20} top/sınıf={top_str} ({time.time()-t0:.0f}s)")

            except Exception as e:
                print(f"  ✗ {mname} seed={run_id} HATA: {e}")

    pd.DataFrame(perclass_rows).to_csv(F_PERCLASS, index=False)
    pd.DataFrame(agree_rows).to_csv(F_AGREE, index=False)
    pd.DataFrame(local_rows).to_csv(F_LOCAL, index=False)
    print(f"\n  ✓ {F_PERCLASS}\n  ✓ {F_AGREE}\n  ✓ {F_LOCAL}")
    write_report(perclass_rows, agree_rows, local_rows)


def make_perclass_figure(sv_by_class, X_explain, feats, mname):
    """Her sınıf için ayrı SHAP bar paneli (#72)."""
    classes = [c for c in sorted(sv_by_class.keys()) if c in CLASS_NAMES]
    if not classes:
        return
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 4.6))
    if len(classes) == 1:
        axes = [axes]
    colors = ["#c44e52", "#dd8452", "#4c72b0"]
    for ax, c in zip(axes, classes):
        imp = np.abs(sv_by_class[c]).mean(axis=0)
        order = np.argsort(imp)
        ax.barh(range(len(order)), imp[order], color=colors[c % 3],
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feats[i] for i in order], fontsize=9)
        ax.set_xlabel("mean |SHAP|", fontsize=10)
        ax.set_title(CLASS_DISPLAY[c], fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Class-specific SHAP importance — {mname}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    safe = mname.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    plt.savefig(os.path.join(FIG_DIR, f"shap_perclass_{safe}.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def write_report(perclass_rows, agree_rows, local_rows):
    pc = pd.DataFrame(perclass_rows)
    ag = pd.DataFrame(agree_rows)
    lo = pd.DataFrame(local_rows)

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\nSINIF-BAZLI SHAP + SIRALAMA UYUMU — ann-thyroid\n")
        f.write("R1-72, R1-73, R1-80, R1-81\n" + "=" * 72 + "\n\n")

        # #72/#73
        f.write("[R1-72/73] SINIF-BAZLI SHAP ÖNEM SIRALAMASI\n")
        f.write("(sınıflar ORTALANMADI — her sınıf ayrı)\n")
        f.write("-" * 72 + "\n")
        if len(pc):
            for mname in pc["model"].unique():
                f.write(f"\n  [{mname}]\n")
                sub = pc[pc.model == mname]
                for cls in sub["class"].unique():
                    s = (sub[sub["class"] == cls]
                         .groupby("feature")["mean_abs_shap"]
                         .agg(["mean", "std"]).round(5)
                         .sort_values("mean", ascending=False))
                    top = ", ".join([f"{i}({r['mean']:.4f})" for i, r in s.head(4).iterrows()])
                    f.write(f"    {cls:<26} top4: {top}\n")

            # sınıflar arası farklılık — #73'ün asıl sorusu
            f.write("\n  → SINIFA ÖZGÜ ÖRÜNTÜ VAR MI? (her sınıfın en önemli özelliği)\n")
            for mname in pc["model"].unique():
                sub = pc[pc.model == mname]
                tops = {}
                for cls in sub["class"].unique():
                    s = sub[sub["class"] == cls].groupby("feature")["mean_abs_shap"].mean()
                    tops[cls] = s.idxmax()
                uniq = len(set(tops.values()))
                verdict = "FARKLI (ortalama bilgi gizler)" if uniq > 1 else "aynı"
                f.write(f"    {mname:<20} {tops} → {verdict}\n")

        # #80
        f.write("\n\n[R1-80] SIRALAMA UYUMU (Spearman ρ, 10 split ort ± std)\n")
        f.write("-" * 72 + "\n")
        if len(ag):
            g = ag.groupby(["model", "ranking_a", "ranking_b"]).agg(
                rho_mean=("spearman_rho", "mean"),
                rho_std=("spearman_rho", "std"),
                p_mean=("p_value", "mean"),
            ).round(4).reset_index()
            f.write(f"{'Model':<20}{'Karşılaştırma':<26}{'Spearman ρ':<18}{'p (ort)':<10}\n")
            for _, r in g.iterrows():
                pair = f"{r['ranking_a']} vs {r['ranking_b']}"
                f.write(f"{r['model']:<20}{pair:<26}"
                        f"{r['rho_mean']:.3f}±{r['rho_std']:.3f}      {r['p_mean']:.4f}\n")
            f.write("\n  → ρ yüksek: HFS'nin seçtiği önem sırası, modelin fiilen\n")
            f.write("    kullandığı önem sırasıyla uyumlu.\n")
            f.write("  → ρ düşük: HFS istatistiksel olarak seçiyor ama model başka\n")
            f.write("    özellikleri kullanıyor (yorumlanabilirlik açısından önemli).\n")

        # #81
        f.write("\n\n[R1-81] YEREL SHAP — TEMSİLİ AZINLIK VAKALAR\n")
        f.write("-" * 72 + "\n")
        if len(lo):
            for mname in lo["model"].unique():
                sub = lo[lo.model == mname]
                f.write(f"\n  [{mname}]\n")
                for (tc, ct), g2 in sub.groupby(["true_class", "case_type"]):
                    pred = g2["pred_class"].iloc[0]
                    top = g2.reindex(g2["shap_contribution"].abs().sort_values(
                        ascending=False).index).head(3)
                    contrib = ", ".join([f"{r['feature']}={r['shap_contribution']:+.4f}"
                                         for _, r in top.iterrows()])
                    f.write(f"    {tc:<26} [{ct:<13}] → tahmin: {pred:<26}\n")
                    f.write(f"      en etkili: {contrib}\n")
            f.write("\n  → Yanlış sınıflanan vakalarda hangi özelliğin modeli yanılttığı\n")
            f.write("    görülebilir; bu, klinik güven için kritik bilgidir.\n")

    print(f"  ✓ {F_REPORT}")

    if len(ag):
        print("\nSIRALAMA UYUMU ÖZETİ (Spearman ρ):")
        g = ag.groupby(["ranking_a", "ranking_b"])["spearman_rho"].mean().round(3)
        for (a, b), v in g.items():
            print(f"  {a:<12} vs {b:<12} ρ={v:+.3f}")


if __name__ == "__main__":
    run_all()
    print("\n" + "=" * 64 + "\nDONE\n" + "=" * 64)
