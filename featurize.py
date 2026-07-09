#!/usr/bin/env python3
"""1-р шат: chronological pass — feature матриц + бөх бүрийн эцсийн төлөв + барилдааны түүх.
TrueSkill-ийг гараар (хурдан closed-form 1v1) хэрэгжүүлэв.
Эх CSV (excel)-ийг шинэчлээд энэ + fit.py-г ажиллуулбал загвар, статистик, түүх бүгд шинэчлэгдэнэ."""
import json, ast, os, time, pickle
from collections import deque
from math import exp, sqrt, erf, pi
import numpy as np, pandas as pd

t0 = time.time()
UPLOADS = "/sessions/determined-nifty-mccarthy/mnt/uploads"   # эх CSV/excel хавтас
OUT = "/sessions/determined-nifty-mccarthy/mnt/outputs/buh_predictor"
TMP = os.path.join(OUT, "_tmp"); os.makedirs(TMP, exist_ok=True)
MATCH_FILES = ["ulsiin buh barildaan.csv", "aimgiin buh barildaan.csv", "sumiin buh barildaan.csv"]
WREST_FILES = ["ulsiin buh.csv", "aimgiin buh.csv", "sumiin buh.csv", "zaan deesh delgerengui.csv"]
K_LIST = [16, 24, 32, 40]; FORM_N = 10; TREND_N = 10; ELO_START = 1500.0

MU0 = 25.0; SIG0 = 25/3.0; BETA = 25/6.0; TAU = 25/300.0
BETA2 = BETA*BETA; TAU2 = TAU*TAU; S2_0 = SIG0*SIG0
_SQRT2 = sqrt(2.0); _INV_SQRT2PI = 1.0/sqrt(2*pi)
def _cdf(x): return 0.5*(1.0+erf(x/_SQRT2))
def _pdf(x): return _INV_SQRT2PI*exp(-0.5*x*x)

W = pd.concat([pd.read_csv(os.path.join(UPLOADS, f), low_memory=False).rename(columns=lambda c: c.replace("﻿", ""))
               for f in WREST_FILES], ignore_index=True)
W["title"] = pd.to_numeric(W["title"], errors="coerce")
W = W.sort_values("title").drop_duplicates("id", keep="last").set_index("id")
def pdob(x):
    if pd.isna(x) or x == "": return np.nan
    try: return pd.Timestamp(x).to_julian_date()
    except Exception: return np.nan
W["dob_jd"] = W["dateOfBirth"].apply(pdob)
W["height_n"] = pd.to_numeric(W["height"], errors="coerce")
W["weight_n"] = pd.to_numeric(W["weight"], errors="coerce")
if "image_local" not in W.columns: W["image_local"] = np.nan
def phist(s):
    if pd.isna(s) or s == "": return []
    try: arr = ast.literal_eval(s)
    except Exception: return []
    out = []
    for d in arr:
        try:
            t = int(d.get("title")); dt = d.get("date") or d.get("fulfilledOn")
            if dt: out.append((pd.Timestamp(dt), t))
        except Exception: pass
    out.sort(); return out
title_hist = {wid: phist(W.at[wid, "titleDocs"]) if "titleDocs" in W.columns else [] for wid in W.index}
TITLE = W["title"].to_dict(); DOB = W["dob_jd"].to_dict(); HT = W["height_n"].to_dict(); WT = W["weight_n"].to_dict()
def title_at(wid, date):
    h = title_hist.get(wid)
    if not h: return TITLE.get(wid, np.nan)
    best = np.nan
    for dt, t in h:
        if dt <= date: best = t if (best != best or t > best) else best
        else: break
    return best if best == best else h[0][1]

mrows = []
for f in MATCH_FILES:
    df = pd.read_csv(os.path.join(UPLOADS, f), low_memory=False).rename(columns=lambda c: c.replace("﻿", ""))
    mrows.append(df[["match_id", "tournament_date", "tournament_name", "tournament_rank", "round", "w1", "w2", "winner", "noShow"]])
M = pd.concat(mrows, ignore_index=True).drop_duplicates("match_id")
M = M[(M["noShow"] != True) & (M["winner"].isin([1, 2]))]
M["date"] = pd.to_datetime(M["tournament_date"], errors="coerce")
M = M.dropna(subset=["date", "w1", "w2"])
M = M[M["w1"] != M["w2"]].sort_values(["date", "match_id"]).reset_index(drop=True)
rnd = pd.to_numeric(M["round"], errors="coerce").fillna(0).values
trk = pd.to_numeric(M["tournament_rank"], errors="coerce").fillna(0).values
dates = list(M["date"]); JD = M["date"].map(lambda d: d.to_julian_date()).values
tname_all = M["tournament_name"].fillna("").astype(str).values
date_all = M["tournament_date"].astype(str).values
print("матч:", len(M), "| ачаалал", round(time.time()-t0, 1), "сек", flush=True)

elo = {k: {} for k in K_LIST}; peak = {k: {} for k in K_LIST}
tmu = {}; ts2 = {}
wins = {}; tot = {}; h2h = {}; form = {}; streak = {}; elo_hist = {}; last_jd = {}; act = {}
hist = {}; tnames = []; tn_idx = {}
w1a, w2a, wina = M["w1"].values, M["w2"].values, M["winner"].values
rng = np.random.default_rng(42); rand = rng.random(len(M))
rows = []; tp = time.time()
for i in range(len(M)):
    a, b, win = w1a[i], w2a[i], wina[i]; d = dates[i]; jd = JD[i]
    if rand[i] < 0.5: A, B, label = b, a, (1 if win == 2 else 0)
    else: A, B, label = a, b, (1 if win == 1 else 0)
    e16 = elo[16]; e24 = elo[24]; e32 = elo[32]; e40 = elo[40]
    muA = tmu.get(A, MU0); muB = tmu.get(B, MU0); s2A = ts2.get(A, S2_0); s2B = ts2.get(B, S2_0)
    ts_wp = _cdf((muA-muB)/sqrt(2*BETA2+s2A+s2B))
    tA, tB = title_at(A, d), title_at(B, d)
    dobA, dobB = DOB.get(A, np.nan), DOB.get(B, np.nan)
    ageA = (jd-dobA)/365.25 if dobA == dobA else np.nan
    ageB = (jd-dobB)/365.25 if dobB == dobB else np.nan
    hA, hB = HT.get(A, np.nan), HT.get(B, np.nan); wtA, wtB = WT.get(A, np.nan), WT.get(B, np.nan)
    nA, nB = tot.get(A, 0), tot.get(B, 0)
    wrA = wins.get(A, 0)/nA if nA else np.nan; wrB = wins.get(B, 0)/nB if nB else np.nan
    fmA = form.get(A); fmB = form.get(B)
    fA = (sum(fmA)/len(fmA)) if fmA else np.nan; fB = (sum(fmB)/len(fmB)) if fmB else np.nan
    ehA, ehB = elo_hist.get(A), elo_hist.get(B)
    trA = (ehA[-1]-ehA[0]) if ehA and len(ehA) > 1 else np.nan
    trB = (ehB[-1]-ehB[0]) if ehB and len(ehB) > 1 else np.nan
    pkA = peak[24].get(A, ELO_START); pkB = peak[24].get(B, ELO_START)
    rustA = (jd-last_jd[A]) if A in last_jd else np.nan
    rustB = (jd-last_jd[B]) if B in last_jd else np.nan
    acA, acB = act.get(A), act.get(B)
    lim = jd-365
    actA = sum(1 for x in acA if x >= lim) if acA else 0
    actB = sum(1 for x in acB if x >= lim) if acB else 0
    key = (A, B) if A < B else (B, A); hh = h2h.get(key, (0, 0))
    h2hA, h2hB = (hh[0], hh[1]) if A < B else (hh[1], hh[0])
    rows.append((e16.get(A, ELO_START)-e16.get(B, ELO_START), e24.get(A, ELO_START)-e24.get(B, ELO_START),
        e32.get(A, ELO_START)-e32.get(B, ELO_START), e40.get(A, ELO_START)-e40.get(B, ELO_START),
        muA-muB, sqrt(s2A)+sqrt(s2B), ts_wp,
        (tA-tB) if (tA == tA and tB == tB) else np.nan,
        (fA-fB) if (fA == fA and fB == fB) else np.nan,
        (wrA-wrB) if (wrA == wrA and wrB == wrB) else np.nan,
        nA-nB, streak.get(A, 0)-streak.get(B, 0),
        (trA-trB) if (trA == trA and trB == trB) else np.nan, pkA-pkB, actA-actB,
        (rustA-rustB) if (rustA == rustA and rustB == rustB) else np.nan,
        (ageA-ageB) if (ageA == ageA and ageB == ageB) else np.nan,
        (hA-hB) if (hA == hA and hB == hB) else np.nan,
        (wtA-wtB) if (wtA == wtA and wtB == wtB) else np.nan,
        h2hA-h2hB, h2hA+h2hB, rnd[i], trk[i], label, jd))
    # ---- update ----
    rw = a if win == 1 else b
    for k in K_LIST:
        ek = elo[k]; pk = peak[k]
        ea, eb = ek.get(a, ELO_START), ek.get(b, ELO_START)
        expa = 1.0/(1.0+10**((eb-ea)/400.0)); sa = 1.0 if win == 1 else 0.0
        na_ = ea+k*(sa-expa); nb_ = eb+k*((1-sa)-(1-expa)); ek[a] = na_; ek[b] = nb_
        if na_ > pk.get(a, ELO_START): pk[a] = na_
        if nb_ > pk.get(b, ELO_START): pk[b] = nb_
    if win == 1: w_, l_ = a, b
    else: w_, l_ = b, a
    muw = tmu.get(w_, MU0); mul = tmu.get(l_, MU0)
    s2w = ts2.get(w_, S2_0)+TAU2; s2l = ts2.get(l_, S2_0)+TAU2
    c2 = 2*BETA2+s2w+s2l; c = sqrt(c2); t = (muw-mul)/c
    cdft = _cdf(t)
    if cdft < 1e-6: cdft = 1e-6
    V = _pdf(t)/cdft; Wt = V*(V+t)
    tmu[w_] = muw+(s2w/c)*V; tmu[l_] = mul-(s2l/c)*V
    ts2[w_] = s2w*(1-(s2w/c2)*Wt); ts2[l_] = s2l*(1-(s2l/c2)*Wt)
    form.setdefault(a, deque(maxlen=FORM_N)).append(1 if win == 1 else 0)
    form.setdefault(b, deque(maxlen=FORM_N)).append(1 if win == 2 else 0)
    for x, won in ((a, win == 1), (b, win == 2)):
        s = streak.get(x, 0); streak[x] = (s+1 if s >= 0 else 1) if won else (s-1 if s <= 0 else -1)
        eh = elo_hist.setdefault(x, deque(maxlen=TREND_N+1)); eh.append(e24[x])
        dq = act.setdefault(x, deque()); dq.append(jd)
        while dq and dq[0] < lim: dq.popleft()
        last_jd[x] = jd
    tot[a] = tot.get(a, 0)+1; tot[b] = tot.get(b, 0)+1
    wins[rw] = wins.get(rw, 0)+1
    kk = (a, b) if a < b else (b, a); r0, r1 = h2h.get(kk, (0, 0))
    h2h[kk] = (r0+1, r1) if rw == kk[0] else (r0, r1+1)
    # барилдааны түүх (профайлд) — tournament нэрийг индексээр хадгална
    tn = tname_all[i]; ds = date_all[i]; rk = int(trk[i])
    ti = tn_idx.get(tn)
    if ti is None: ti = len(tnames); tnames.append(tn); tn_idx[tn] = ti
    hist.setdefault(a, []).append((ds, ti, rk, b, 1 if win == 1 else 0))
    hist.setdefault(b, []).append((ds, ti, rk, a, 1 if win == 2 else 0))

cols = [f"elo_diff_{k}" for k in K_LIST] + ["ts_mu_diff", "ts_sigma_sum", "ts_wp", "title_diff",
    "form_diff", "winrate_diff", "experience_diff", "streak_diff", "elo_trend_diff", "peak_elo_diff",
    "activity_diff", "rust_diff", "age_diff", "height_diff", "weight_diff", "h2h_diff", "h2h_total",
    "round_n", "trank_n", "label", "jd"]
D = pd.DataFrame(rows, columns=cols)
D.to_pickle(os.path.join(TMP, "D.pkl"))
state = {"elo": {k: elo[k] for k in K_LIST}, "peak": {k: peak[k] for k in K_LIST},
    "ts": {w: (tmu[w], sqrt(ts2[w])) for w in tmu}, "wins": wins, "tot": tot,
    "form": {w: (sum(dq)/len(dq)) for w, dq in form.items()}, "streak": streak,
    "elo_trend": {w: (eh[-1]-eh[0] if len(eh) > 1 else 0.0) for w, eh in elo_hist.items()},
    "last_jd": last_jd, "act": {w: list(dq) for w, dq in act.items()}}
with open(os.path.join(TMP, "state.pkl"), "wb") as f: pickle.dump(state, f)
h2h_out = {f"{k[0]}|{k[1]}": [v[0], v[1]] for k, v in h2h.items() if (v[0]+v[1]) > 0}
with open(os.path.join(TMP, "h2h.json"), "w", encoding="utf-8") as f: json.dump(h2h_out, f, ensure_ascii=False)
with open(os.path.join(TMP, "history.pkl"), "wb") as f: pickle.dump({"tnames": tnames, "hist": hist}, f)
W[["firstName", "lastName", "title", "title_name", "dob_jd", "height_n", "weight_n", "image_local"]].to_pickle(os.path.join(TMP, "W.pkl"))
print("Дууслаа. pass", round(time.time()-tp, 1), "сек | нийт", round(time.time()-t0, 1), "сек | D", D.shape, "| tnames", len(tnames), flush=True)
