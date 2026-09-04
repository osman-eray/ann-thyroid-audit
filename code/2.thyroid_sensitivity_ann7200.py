# ============================================================
# SCRIPT 2 — HFS Parametre Duyarlılık Analizi
# Thyroid Disease (ann-thyroid, 7200 instances, 3-class)
#
# Hakem yorumları: R1 (sensitivity analysis talebi) · genel metodolojik sağlamlık
#
# NE YAPAR:
#   HFS'nin üç ana parametresi üzerinde tek-değişkenli (one-at-a-time) tarama:
#     k     = seçilen özellik sayısı   {4, 6, 8, 10, 12}      (varsayılan 8)
#     w     = IG/GR ağırlığı           {0.0, 0.25, 0.5, 0.75, 1.0}  (varsayılan 0.5)
#     theta = korelasyon eşiği         {0.80, 0.85, 0.90, 0.95, 0.99} (varsayılan 0.90)
#   Her parametre değeri için HFS+SMOTE pipeline'ı 10 bağımsız split üzerinde koşar,
#   temsili bir model seti (LR, RF, XGB) ile ROC-AUC / Macro-F1 / Specificity raporlar.
#   Bir parametre taranırken diğer ikisi varsayılanda sabit tutulur.
#
# ÇIKTI:
#   sensitivity_k.csv, sensitivity_w.csv, sensitivity_theta.csv
#   sensitivity_summary.csv  (tüm taramalar birleşik)
#   sensitivity_report.txt
#
# RESUME: (param, değer, seed) üçlüsü bazında; kesintide kaldığı yerden devam.
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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10
SPLIT_SEEDS = list(range(N_RUNS))
INTERNAL_SEED = 42

DATA_PATH  = "thyroid_ann_7200.csv"
OUTPUT_DIR = "thyroid_sensitivity_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_SUMMARY = os.path.join(OUTPUT_DIR, "sensitivity_summary.csv")
F_REPORT  = os.path.join(OUTPUT_DIR, "sensitivity_report.txt")

FEATURE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]
TARGET_COL = 'target'
CLASS_NAMES = {0: "hyperthyroid", 1: "subnormal", 2: "normal"}

# Varsayılan HFS parametreleri (makale ile aynı)
DEF_THETA = 0.90
DEF_W     = 0.50
DEF_K     = 8

# Tarama gridleri
GRID_K     = [4, 6, 8, 10, 12]
GRID_W     = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID_THETA = [0.80, 0.85, 0.90, 0.95, 0.99]

# Temsili modeller (tam 10 model yerine hız için 3 temsilci; duyarlılık trendi için yeterli)
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=INTERNAL_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=INTERNAL_SEED),
        "XGBoost":             XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',
                                             random_state=INTERNAL_SEED),
    }

PARAM_GRIDS_SMOTE = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [10, None]},
    "XGBoost":             {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
}
SCORING_CV = 'roc_auc_ovr_weighted'

# ============================================================
# HFS (parametrik — theta, w, k dışarıdan verilir)
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
        counts = np.histogram(X_col, bins=n_bins)[0]
        counts = counts[counts > 0]
    probs = counts / counts.sum()
    Hx = -np.sum(probs * np.log2(probs))
    return IG / (Hx + 1e-12) if Hx > 0 else 0.0

def hybrid_feature_selection(df, target_col, theta, w, k):
    features = [c for c in df.columns if c != target_col]
    X = df[features].values
    y = df[target_col].values
    # Stage 1: Redundancy elimination
    redundant = set()
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > theta:
                redundant.add(features[j])
    non_redundant = [f for f in features if f not in redundant]
    # Stage 2-3: IG, GR, hybrid score
    scores = []
    for f in non_redundant:
        X_f = df[f].values
        ig = information_gain(X_f, y)
        gr = gain_ratio(X_f, y)
        hs = w * ig + (1 - w) * gr
        scores.append((f, hs))
    scores.sort(key=lambda x: x[1], reverse=True)
    k_eff = min(k, len(scores))
    return [f for f, _ in scores[:k_eff]]

# ============================================================
# Değerlendirme (tek param değeri, tek seed)
# ============================================================

def macro_specificity(cm, labels):
    total = cm.sum()
    specs = []
    for idx in range(len(labels)):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specs)

def eval_one(run_id, data, theta, w, k):
    X_raw = data[FEATURE_COLS]
    y = data[TARGET_COL] - 1
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.2, random_state=run_id, stratify=y)
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), columns=FEATURE_COLS)
    Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=FEATURE_COLS)
    # HFS (train-fold içinde, leakage-free)
    dfh = Xtr.copy(); dfh['target'] = ytr.values
    feats = hybrid_feature_selection(dfh, 'target', theta=theta, w=w, k=k)
    XtrH, XteH = Xtr[feats], Xte[feats]
    labels = sorted(yte.unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=INTERNAL_SEED)
    rows = []
    for name, model in get_models().items():
        pipe = ImbPipeline([("smote", SMOTE(random_state=INTERNAL_SEED)), ("model", model)])
        grid = GridSearchCV(pipe, PARAM_GRIDS_SMOTE[name], cv=cv, scoring=SCORING_CV, n_jobs=-1)
        grid.fit(XtrH, ytr)
        best = grid.best_estimator_
        yprob = best.predict_proba(XteH)
        ypred = best.predict(XteH)
        roc = roc_auc_score(yte, yprob, multi_class='ovr', average='weighted')
        f1m = f1_score(yte, ypred, average='macro', zero_division=0)
        cm = confusion_matrix(yte, ypred, labels=labels)
        spec = macro_specificity(cm, labels)
        rows.append({"model": name, "n_selected": len(feats),
                     "roc_auc_ovr": roc, "f1_macro": f1m, "specificity_macro": spec})
    return rows

# ============================================================
# Resume
# ============================================================

def load_done():
    if not os.path.exists(F_SUMMARY):
        return set()
    df = pd.read_csv(F_SUMMARY)
    return set(zip(df["param"], df["value"].astype(str), df["run_id"], df["model"]))

def append_csv(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ============================================================
# Tarama motoru
# ============================================================

def sweep(param_name, grid, data, done):
    print(f"\n{'='*60}\nTARAMA: {param_name}  değerler={grid}\n{'='*60}")
    for val in grid:
        for run_id in SPLIT_SEEDS:
            key = (param_name, str(val), run_id)
            # her modeli tek tek kontrol etmek yerine seed bazında bak
            if all((param_name, str(val), run_id, m) in done for m in get_models().keys()):
                continue
            # parametre setini kur
            theta, w, k = DEF_THETA, DEF_W, DEF_K
            if param_name == "k":     k = int(val)
            if param_name == "w":     w = float(val)
            if param_name == "theta": theta = float(val)
            t0 = time.time()
            try:
                rows = eval_one(run_id, data, theta=theta, w=w, k=k)
            except Exception as e:
                print(f"  ✗ {param_name}={val} seed={run_id} HATA: {e}")
                continue
            for r in rows:
                r.update({"param": param_name, "value": val, "run_id": run_id})
            append_csv(rows, F_SUMMARY)
            print(f"  {param_name}={val} seed={run_id}: "
                  f"ROC={np.mean([r['roc_auc_ovr'] for r in rows]):.4f} "
                  f"({time.time()-t0:.0f}s)")

# ============================================================
# Aggregate + rapor
# ============================================================

def aggregate():
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
    df = pd.read_csv(F_SUMMARY)
    # her (param, value, model) için mean±std
    agg = df.groupby(["param", "value", "model"]).agg(
        roc_mean=("roc_auc_ovr", "mean"), roc_std=("roc_auc_ovr", "std"),
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        spec_mean=("specificity_macro", "mean"), spec_std=("specificity_macro", "std"),
        n_sel=("n_selected", "mean"),
    ).round(4).reset_index()
    # ayrı CSV'ler
    for p in ["k", "w", "theta"]:
        sub = agg[agg["param"] == p]
        if len(sub):
            sub.to_csv(os.path.join(OUTPUT_DIR, f"sensitivity_{p}.csv"), index=False)
            print(f"  ✓ sensitivity_{p}.csv")
    # rapor
    with open(F_REPORT, "w", encoding="utf-8") as f:
        f.write("="*70 + "\nHFS PARAMETRE DUYARLILIK ANALİZİ — ann-thyroid\n")
        f.write(f"Varsayılan: k={DEF_K}, w={DEF_W}, theta={DEF_THETA}\n" + "="*70 + "\n\n")
        for p, grid, default in [("k", GRID_K, DEF_K), ("w", GRID_W, DEF_W), ("theta", GRID_THETA, DEF_THETA)]:
            f.write(f"\n[{p}] tarama (diğerleri varsayılanda sabit)\n" + "-"*70 + "\n")
            sub = agg[agg["param"] == p]
            # modeller ortalaması ile tek satır trend
            for val in grid:
                vv = sub[sub["value"] == val]
                if len(vv):
                    f.write(f"  {p}={val:<6} ROC={vv['roc_mean'].mean():.4f}  "
                            f"Macro-F1={vv['f1_mean'].mean():.4f}  "
                            f"Spec={vv['spec_mean'].mean():.4f}  "
                            f"(seçilen≈{vv['n_sel'].mean():.1f})"
                            f"{'  ← varsayılan' if val==default else ''}\n")
    print(f"  ✓ {F_REPORT}")
    print("\nÖzet (model-ortalaması ROC):")
    for p, grid in [("k", GRID_K), ("w", GRID_W), ("theta", GRID_THETA)]:
        sub = agg[agg["param"] == p]
        trend = [f"{v}:{sub[sub['value']==v]['roc_mean'].mean():.3f}" for v in grid if len(sub[sub['value']==v])]
        print(f"  {p:<6} {trend}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("HFS DUYARLILIK ANALİZİ — ann-thyroid")
    data = pd.read_csv(DATA_PATH)
    print(f"Data: {DATA_PATH} | {len(data)} örnek, {len(FEATURE_COLS)} özellik")
    print(f"Sınıflar: {data[TARGET_COL].value_counts().sort_index().to_dict()}")
    print("="*60)
    done = load_done()
    if done:
        print(f"[RESUME] {len(done)} (param,değer,seed,model) zaten tamam")
    sweep("k",     GRID_K,     data, done)
    sweep("w",     GRID_W,     data, done)
    sweep("theta", GRID_THETA, data, done)
    aggregate()
    print("\n" + "="*60 + "\nDONE\n" + "="*60)
