# ============================================================
# SCRIPT 4 — SHAP Açıklanabilirlik Stabilite Analizi
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1 (interpretability robustness) · açıklanabilirlik güvenilirliği
#
# NE YAPAR:
#   HFS+SMOTE pipeline'ı 10 bağımsız split üzerinde koşulurken, her run için
#   SHAP tabanlı özellik önem sıralaması çıkarılır. Sonra bu 10 sıralamanın
#   ne kadar KARARLI (stable) olduğu ölçülür:
#     - Kendall's W (concordance)        → sıralama uyumu (0=rastgele, 1=tam uyum)
#     - Ortalama Spearman ρ (çiftler arası)
#     - Top-k özellik seçim frekansı     → hangi özellikler tutarlı biçimde en önemli
#     - Ortalama |SHAP| değeri ± std     → her özellik için önem büyüklüğü kararlılığı
#
#   SHAP hesaplama modele göre:
#     - TreeExplainer: RandomForest, XGBoost, DecisionTree
#     - LinearExplainer: Logistic Regression
#   Çok-sınıf SHAP (n×feat×class) → sınıflar arası |SHAP| ortalaması ile tek skor.
#
# ÇIKTI:
#   shap_importance_all_runs.csv, shap_stability_summary.csv,
#   shap_rank_frequency.csv, shap_stability_report.txt
#
# RESUME: (model, seed) bazında.
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
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

try:
    import shap
except ImportError:
    raise SystemExit("shap gerekli: pip install shap")

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42
K_SELECT = 8
SHAP_SAMPLE = 300   # SHAP hesaplaması için test örneklem boyutu (hız/kararlılık dengesi)

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_shap_stability_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_ALL      = os.path.join(OUTPUT_DIR, "shap_importance_all_runs.csv")
F_SUMMARY  = os.path.join(OUTPUT_DIR, "shap_stability_summary.csv")
F_RANKFREQ = os.path.join(OUTPUT_DIR, "shap_rank_frequency.csv")
F_REPORT   = os.path.join(OUTPUT_DIR, "shap_stability_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'

# SHAP-uyumlu modeller (Tree/Linear explainer olanlar)
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
# HFS (makale ile birebir)
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
    df = Xtr.copy()
    feats = list(df.columns)
    X = df.values
    redundant = set()
    n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                redundant.add(feats[j])
    non_red = [f for f in feats if f not in redundant]
    scores = []
    for f in non_red:
        xf = df[f].values
        hs = w * information_gain(xf, ytr) + (1 - w) * gain_ratio(xf, ytr)
        scores.append((f, hs))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores[:min(k, len(scores))]]

# ============================================================
# SHAP önem çıkarımı (model tipine göre)
# ============================================================

def shap_importance(model, X_background, X_explain, feat_names, model_name):
    """Her özellik için ortalama |SHAP| döndürür (çok-sınıf: sınıflar arası ort)."""
    try:
        if model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_background)
            sv = explainer.shap_values(X_explain)
        else:  # Tree tabanlı
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_explain)
    except Exception:
        # genel fallback
        explainer = shap.Explainer(model, X_background)
        sv = explainer(X_explain).values

    sv = np.array(sv)
    # Şekil normalizasyonu → (n, feat) mutlak ortalama
    if sv.ndim == 3:
        # (n, feat, class) veya (class, n, feat)
        if sv.shape[-1] == len(feat_names):        # (class, n, feat)
            imp = np.abs(sv).mean(axis=(0, 1))
        else:                                       # (n, feat, class)
            imp = np.abs(sv).mean(axis=(0, 2))
    elif isinstance(sv, list):
        imp = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
    else:
        imp = np.abs(sv).mean(0)
    imp = np.asarray(imp).ravel()[:len(feat_names)]
    return dict(zip(feat_names, imp))

# ============================================================
# Kendall's W
# ============================================================

def kendalls_w(rank_matrix):
    """rank_matrix: (n_raters × n_items) sıralama matrisi. W ∈ [0,1]."""
    m, n = rank_matrix.shape          # m rater (run), n item (feature)
    R = rank_matrix.sum(axis=0)        # her item'ın toplam sırası
    Rbar = R.mean()
    S = np.sum((R - Rbar) ** 2)
    denom = m**2 * (n**3 - n)
    return 12 * S / denom if denom > 0 else 0.0

# ============================================================
# Resume
# ============================================================

def load_done():
    if not os.path.exists(F_ALL):
        return set()
    df = pd.read_csv(F_ALL)
    return set(zip(df["model"], df["run_id"]))

def append_csv(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ============================================================
# Ana döngü
# ============================================================

def run_all():
    print("="*60)
    print("SHAP STABİLİTE ANALİZİ — ann-thyroid")
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {len(data)} örnek | SHAP örneklem: {SHAP_SAMPLE}")
    print(f"Modeller: {list(get_models().keys())}")
    print("="*60)
    done = load_done()
    if done:
        print(f"[RESUME] {len(done)} (model,seed) tamam")

    for run_id in SPLIT_SEEDS:
        X_raw = data[FEATURE_COLS]; y = data[TARGET_COL] - 1
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X_raw, y, test_size=0.2, random_state=run_id, stratify=y)
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)
        feats = hfs_select(Xtr, ytr.values)
        XtrH, XteH = Xtr[feats], Xte[feats]

        # SHAP açıklama örneklemi
        n_ex = min(SHAP_SAMPLE, len(XteH))
        rng = np.random.RandomState(INTERNAL_SEED + run_id)
        ex_idx = rng.choice(len(XteH), n_ex, replace=False)
        X_explain = XteH.iloc[ex_idx]
        X_bg = XtrH.sample(min(100, len(XtrH)), random_state=INTERNAL_SEED)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
        for mname, model in get_models().items():
            if (mname, run_id) in done:
                continue
            t0 = time.time()
            try:
                pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)), ("model", model)])
                grid = GridSearchCV(pipe, PARAM_GRIDS[mname], cv=cv, scoring=SCORING_CV, n_jobs=-1)
                grid.fit(XtrH, ytr)
                fitted = grid.best_estimator_.named_steps["model"]
                imp = shap_importance(fitted, X_bg, X_explain, feats, mname)
            except Exception as e:
                print(f"  ✗ {mname} seed={run_id} HATA: {e}")
                continue
            rows = [{"model": mname, "run_id": run_id, "feature": f,
                     "mean_abs_shap": float(imp.get(f, 0.0))} for f in feats]
            append_csv(rows, F_ALL)
            top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  seed={run_id} {mname:<18} top3={[t[0] for t in top]} ({time.time()-t0:.0f}s)")
    aggregate()

def aggregate():
    print(f"\n{'='*60}\nAGGREGATE — Stabilite\n{'='*60}")
    df = pd.read_csv(F_ALL)
    summ_rows, freq_rows = [], []

    for mname, g in df.groupby("model"):
        runs = sorted(g["run_id"].unique())
        feats = sorted(g["feature"].unique())
        # her run için özellik→önem sözlüğü, sonra sıralama
        rank_mat = []
        imp_by_feat = {f: [] for f in feats}
        for r in runs:
            gr = g[g.run_id == r].set_index("feature")["mean_abs_shap"]
            # eksik özellikleri 0 ile doldur (HFS runlar arası farklı seçebilir)
            vals = np.array([gr.get(f, 0.0) for f in feats])
            for f, v in zip(feats, vals):
                imp_by_feat[f].append(v)
            # sıralama: yüksek önem = düşük rank numarası (1 en önemli)
            order = np.argsort(vals)[::-1]
            ranks = np.empty(len(feats))
            ranks[order] = np.arange(1, len(feats) + 1)
            rank_mat.append(ranks)
        rank_mat = np.array(rank_mat)

        W = kendalls_w(rank_mat) if len(runs) > 1 else np.nan
        # çiftler arası ortalama Spearman
        rhos = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                rho, _ = spearmanr(rank_mat[i], rank_mat[j])
                if not np.isnan(rho):
                    rhos.append(rho)
        mean_rho = np.mean(rhos) if rhos else np.nan

        summ_rows.append({"model": mname, "n_runs": len(runs),
                          "kendall_w": round(W, 4) if not np.isnan(W) else np.nan,
                          "mean_spearman": round(mean_rho, 4) if not np.isnan(mean_rho) else np.nan})

        # top-3 seçim frekansı
        for f in feats:
            top3_count = 0
            for r in runs:
                gr = g[g.run_id == r].set_index("feature")["mean_abs_shap"]
                vals = pd.Series({ff: gr.get(ff, 0.0) for ff in feats})
                if f in vals.nlargest(3).index:
                    top3_count += 1
            freq_rows.append({"model": mname, "feature": f,
                              "mean_abs_shap": round(np.mean(imp_by_feat[f]), 5),
                              "std_abs_shap": round(np.std(imp_by_feat[f]), 5),
                              "top3_frequency": top3_count, "n_runs": len(runs)})

    pd.DataFrame(summ_rows).to_csv(F_SUMMARY, index=False)
    print(f"  ✓ {F_SUMMARY}")
    pd.DataFrame(freq_rows).sort_values(["model", "mean_abs_shap"], ascending=[True, False]) \
        .to_csv(F_RANKFREQ, index=False)
    print(f"  ✓ {F_RANKFREQ}")

    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("="*70 + "\nSHAP STABİLİTE ANALİZİ — ann-thyroid\n")
        f.write("Kendall's W: 1.0=tam sıralama uyumu, 0=rastgele\n" + "="*70 + "\n\n")
        f.write(f"{'Model':<20}{'Kendall W':<14}{'Mean Spearman':<16}{'#runs':<8}\n" + "-"*70 + "\n")
        for r in summ_rows:
            f.write(f"{r['model']:<20}{str(r['kendall_w']):<14}{str(r['mean_spearman']):<16}{r['n_runs']:<8}\n")
        f.write("\n\nEN KARARLI TOP ÖZELLİKLER (model başına, top3 frekansı):\n" + "-"*70 + "\n")
        fr = pd.DataFrame(freq_rows)
        for mname in fr["model"].unique():
            sub = fr[fr.model == mname].nlargest(5, "mean_abs_shap")
            f.write(f"\n  [{mname}]\n")
            for _, r in sub.iterrows():
                f.write(f"    {r['feature']:<18} |SHAP|={r['mean_abs_shap']:.4f}±{r['std_abs_shap']:.4f}  "
                        f"top3: {r['top3_frequency']}/{r['n_runs']}\n")
    print(f"  ✓ {F_REPORT}")
    print("\nStabilite özeti (Kendall's W):")
    for r in summ_rows:
        print(f"  {r['model']:<20} W={r['kendall_w']}  Spearman={r['mean_spearman']}")

if __name__ == "__main__":
    run_all()
    print("\n" + "="*60 + "\nDONE\n" + "="*60)
