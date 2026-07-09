#!/usr/bin/env python3
"""2-р шат: K сонголт → загвар tune (HistGradientBoosting) → isotonic калибровк → артефакт.

Тайлбар: Эцсийн загвар нь sklearn-ий HistGradientBoosting — цэвэр Python, ямар ч нэмэлт
native сан (жишээ нь LightGBM-ийн libomp) шаардахгүй тул хаана ч ажиллана.
Хэрэв LightGBM туршихыг хүсвэл USE_LGBM=True болгоно (macOS: `brew install libomp`)."""
import json, os, pickle, time
from datetime import datetime
import numpy as np, pandas as pd, joblib

USE_LGBM = False

t0 = time.time()
OUT = "/sessions/determined-nifty-mccarthy/mnt/outputs/buh_predictor"
TMP = os.path.join(OUT, "_tmp"); ART = os.path.join(OUT, "artifacts"); os.makedirs(ART, exist_ok=True)
ELO_START = 1500.0; BETA = 25/6.0; BETA2 = BETA*BETA
K_LIST = [16, 24, 32, 40]

D = pd.read_pickle(os.path.join(TMP, "D.pkl")).sort_values("jd").reset_index(drop=True)
with open(os.path.join(TMP, "state.pkl"), "rb") as f: state = pickle.load(f)
Wp = pd.read_pickle(os.path.join(TMP, "W.pkl"))
with open(os.path.join(TMP, "h2h.json"), encoding="utf-8") as f: H2H = json.load(f)

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
if USE_LGBM:
    import lightgbm as lgb

n = len(D); i70, i85 = int(n*0.70), int(n*0.85)

va0 = D.iloc[i70:i85]
bestK, bestAUC = 24, 0.0
print(">> Elo K сонголт:")
for k in K_LIST:
    prob = 1.0/(1.0+10**(-va0[f"elo_diff_{k}"]/400.0))
    au = roc_auc_score(va0["label"], prob); print(f"   K={k}: AUC {au:.4f}")
    if au > bestAUC: bestAUC, bestK = au, k
print("   -> K =", bestK)
D["elo_diff"] = D[f"elo_diff_{bestK}"]

FEATS = ["elo_diff", "ts_mu_diff", "ts_sigma_sum", "ts_wp", "title_diff", "form_diff",
         "winrate_diff", "experience_diff", "streak_diff", "elo_trend_diff", "peak_elo_diff",
         "activity_diff", "rust_diff", "age_diff", "height_diff", "weight_diff",
         "h2h_diff", "h2h_total", "round_n", "trank_n"]
tr, va, te = D.iloc[:i70], D.iloc[i70:i85], D.iloc[i85:]
Xtr, ytr = tr[FEATS], tr["label"]; Xva, yva = va[FEATS], va["label"]; Xte, yte = te[FEATS], te["label"]

def au_of(m, X, y): return roc_auc_score(y, m.predict_proba(X)[:, 1])

def make_hgb(lr, lv):
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=lr, max_leaf_nodes=lv,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
        validation_fraction=0.1, early_stopping=True)

print(">> Загвар tune (validation) ...")
cands = []
for lr in [0.03, 0.05, 0.08]:
    for lv in [31, 63]:
        m = make_hgb(lr, lv).fit(Xtr, ytr)
        cands.append(("HistGBM", {"lr": lr, "leaves": lv}, au_of(m, Xva, yva), m))
if USE_LGBM:
    for lr in [0.03, 0.05]:
        for lv in [31, 63]:
            m = lgb.LGBMClassifier(n_estimators=800, learning_rate=lr, num_leaves=lv, min_child_samples=40,
                reg_lambda=1.0, subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1).fit(Xtr, ytr)
            cands.append(("LightGBM", {"lr": lr, "leaves": lv}, au_of(m, Xva, yva), m))
cands.sort(key=lambda c: -c[2])
for nm, pr, au, _ in cands[:6]: print(f"   {nm} {pr}: val AUC {au:.4f}")
best_name, best_prm, best_val, best_model = cands[0]
print(f"   -> шилдэг: {best_name} {best_prm}")

cal = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit").fit(Xva, yva)
pu = best_model.predict_proba(Xte)[:, 1]; pc = cal.predict_proba(Xte)[:, 1]
acc_u = accuracy_score(yte, (pu >= .5)); acc_c = accuracy_score(yte, (pc >= .5))
auc_u = roc_auc_score(yte, pu); auc_c = roc_auc_score(yte, pc)
ll_u = log_loss(yte, pu); ll_c = log_loss(yte, pc)
base = accuracy_score(yte[te["title_diff"] != 0], (te["title_diff"] > 0)[te["title_diff"] != 0])
elo_only = accuracy_score(yte, (1/(1+10**(-te["elo_diff"]/400)) >= .5))
print("\n===== ҮР ДҮН (эцсийн 15% test) =====")
print(f"v3 {best_name} калибровкгүй: acc {acc_u:.4f} | AUC {auc_u:.4f} | logloss {ll_u:.4f}")
print(f"v3 {best_name} калибровктой: acc {acc_c:.4f} | AUC {auc_c:.4f} | logloss {ll_c:.4f}")
print(f"Зөвхөн Elo(K={bestK}): acc {elo_only:.4f} | Зөвхөн цол: acc {base:.4f}")

from sklearn.inspection import permutation_importance
imp = permutation_importance(best_model, Xte.fillna(Xte.median()), yte, n_repeats=1, random_state=0, scoring="roc_auc")
print("\nTop feature (AUC drop):")
for nm, mv in sorted(zip(FEATS, imp.importances_mean), key=lambda x: -x[1])[:10]:
    print(f"   {nm:16s} {mv:.4f}")

trva = D.iloc[:i85]
fm = make_hgb(best_prm["lr"], best_prm["leaves"])
fm.fit(trva[FEATS], trva["label"])
fcal = CalibratedClassifierCV(fm, method="isotonic", cv="prefit").fit(te[FEATS], te["label"])
joblib.dump({"model": fcal, "feats": FEATS, "bestK": bestK, "beta2": BETA2}, os.path.join(ART, "model.joblib"))

today_jd = pd.Timestamp(datetime.utcnow().date()).to_julian_date()
elo_b = state["elo"][bestK]; peak_b = state["peak"][bestK]
tsD = state["ts"]; wins = state["wins"]; tot = state["tot"]; formD = state["form"]
strk = state["streak"]; trend = state["elo_trend"]; lastj = state["last_jd"]; actD = state["act"]
players = []
for wid, nn in tot.items():
    if nn == 0 or wid not in Wp.index: continue
    ww = wins.get(wid, 0)
    dob = Wp.at[wid, "dob_jd"]; age = (today_jd-dob)/365.25 if dob == dob else None
    mu, sig = tsD.get(wid, (25.0, 25/3.0))
    ac = actD.get(wid, []); act365 = sum(1 for x in ac if x >= today_jd-365)
    players.append({
        "id": wid,
        "firstName": "" if pd.isna(Wp.at[wid, "firstName"]) else str(Wp.at[wid, "firstName"]),
        "lastName": "" if pd.isna(Wp.at[wid, "lastName"]) else str(Wp.at[wid, "lastName"]),
        "title": None if pd.isna(Wp.at[wid, "title"]) else int(Wp.at[wid, "title"]),
        "title_name": "" if pd.isna(Wp.at[wid, "title_name"]) else str(Wp.at[wid, "title_name"]),
        "age": None if (age is None or age != age) else round(float(age), 1),
        "height": None if pd.isna(Wp.at[wid, "height_n"]) else float(Wp.at[wid, "height_n"]),
        "weight": None if pd.isna(Wp.at[wid, "weight_n"]) else float(Wp.at[wid, "weight_n"]),
        "wins": int(ww), "matches": int(nn), "winrate": round(ww/nn, 3) if nn else None,
        "elo": round(float(elo_b.get(wid, ELO_START)), 1),
        "peak_elo": round(float(peak_b.get(wid, ELO_START)), 1),
        "ts_mu": round(float(mu), 3), "ts_sigma": round(float(sig), 3),
        "form": None if wid not in formD else round(float(formD[wid]), 3),
        "streak": int(strk.get(wid, 0)), "elo_trend": round(float(trend.get(wid, 0.0)), 1),
        "activity365": int(act365), "last_match_jd": round(float(lastj.get(wid, today_jd)), 1),
        "img": (os.path.basename(str(Wp.at[wid, "image_local"]))
                if ("image_local" in Wp.columns and not pd.isna(Wp.at[wid, "image_local"]) and str(Wp.at[wid, "image_local"]))
                else None),
    })
players.sort(key=lambda x: -x["elo"])
with open(os.path.join(ART, "players.json"), "w", encoding="utf-8") as f:
    json.dump(players, f, ensure_ascii=False)
with open(os.path.join(ART, "h2h.json"), "w", encoding="utf-8") as f:
    json.dump(H2H, f, ensure_ascii=False)

# барилдааны түүхийг артефакт руу зөөнө (профайлд)
import shutil
shutil.copy(os.path.join(TMP, "history.pkl"), os.path.join(ART, "history.pkl"))

metrics = {"accuracy": round(float(acc_c), 4), "auc": round(float(auc_c), 4), "log_loss": round(float(ll_c), 4),
    "accuracy_uncal": round(float(acc_u), 4), "elo_only_acc": round(float(elo_only), 4),
    "baseline_acc": round(float(base), 4), "bestK": bestK, "model": best_name,
    "n_test": int(len(te)), "n_matches": int(n), "n_players": len(players),
    "trained_at": datetime.utcnow().isoformat(), "history": {"v1": 0.737, "v2": 0.760}}
with open(os.path.join(ART, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\n>> Дууслаа. {best_name} | test acc {acc_c:.4f} | {round(time.time()-t0,1)} сек | бөх {len(players)}")
