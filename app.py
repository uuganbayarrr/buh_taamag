#!/usr/bin/env python3
"""
Монгол бөхийн таамаглах вэб апп (FastAPI, v4) — 3 модуль:
  1) 1v1 таамаглал  2) Наадам bracket (гараар 16 оруулж болно)  3) Бөхийн профайл (мэдээлэл+түүх+зураг)
Зураг: BUH_IMAGES орчны хувьсагчид "zaan deesh zurag" фолдерын замыг зааж өгнө
       (эсвэл ./images дотор <id>.jpg байршуулна).
Ажиллуулах:  python3 -m uvicorn app:app --reload  →  http://127.0.0.1:8000
"""
import os, json, pickle
from math import sqrt, erf, log2
from datetime import datetime
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from typing import Optional, List

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
IMAGES_DIR = os.environ.get("BUH_IMAGES", os.path.join(HERE, "images"))

bundle = joblib.load(os.path.join(ART, "model.joblib"))
MODEL, FEATS, BETA2 = bundle["model"], bundle["feats"], bundle.get("beta2", (25/6.0) ** 2)
with open(os.path.join(ART, "players.json"), encoding="utf-8") as f:
    PLAYERS = json.load(f)
with open(os.path.join(ART, "h2h.json"), encoding="utf-8") as f:
    H2H = json.load(f)
with open(os.path.join(ART, "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)
try:
    with open(os.path.join(ART, "history.pkl"), "rb") as f:
        HIST = pickle.load(f)
except Exception:
    HIST = {"tnames": [], "hist": {}}

PMAP = {p["id"]: p for p in PLAYERS}
_epoch = datetime(2000, 1, 1)
TODAY_JD = 2451544.5 + (datetime.utcnow() - _epoch).total_seconds() / 86400.0
ACTIVE = [p for p in PLAYERS if (TODAY_JD - p.get("last_match_jd", 0)) <= 730]
# Elo зэрэглэл (профайлд)
_rank = {p["id"]: i + 1 for i, p in enumerate(sorted(PLAYERS, key=lambda x: -(x.get("elo") or 0)))}

app = FastAPI(title="Бөх таамаглагч v4")


def _cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def full_name(p):
    return ((p.get("lastName") or "").strip() + " " + (p.get("firstName") or "").strip()).strip() or p["id"]


NAME = {p["id"]: full_name(p) for p in PLAYERS}


def build_features(A, B):
    def g(p, k):
        v = p.get(k)
        return np.nan if v is None else float(v)

    a, b = A["id"], B["id"]
    if a < b:
        rec = H2H.get(f"{a}|{b}", [0, 0]); h2hA, h2hB = rec[0], rec[1]
    else:
        rec = H2H.get(f"{b}|{a}", [0, 0]); h2hA, h2hB = rec[1], rec[0]
    muA, muB = g(A, "ts_mu"), g(B, "ts_mu")
    sgA, sgB = g(A, "ts_sigma"), g(B, "ts_sigma")
    ts_wp = _cdf((muA - muB) / sqrt(2 * BETA2 + sgA ** 2 + sgB ** 2))
    rust_diff = (TODAY_JD - g(A, "last_match_jd")) - (TODAY_JD - g(B, "last_match_jd"))
    row = [
        g(A, "elo") - g(B, "elo"), muA - muB, sgA + sgB, ts_wp,
        g(A, "title") - g(B, "title"), g(A, "form") - g(B, "form"),
        g(A, "winrate") - g(B, "winrate"), A["matches"] - B["matches"],
        g(A, "streak") - g(B, "streak"), g(A, "elo_trend") - g(B, "elo_trend"),
        g(A, "peak_elo") - g(B, "peak_elo"), g(A, "activity365") - g(B, "activity365"),
        rust_diff, g(A, "age") - g(B, "age"), g(A, "height") - g(B, "height"),
        g(A, "weight") - g(B, "weight"), h2hA - h2hB, h2hA + h2hB, 1.0, 1.0,
    ]
    return np.array([row], dtype=float), (h2hA, h2hB)


def winprob_matrix(field):
    m = len(field)
    rows, idx = [], []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            X, _ = build_features(field[i], field[j])
            rows.append(X[0]); idx.append((i, j))
    probs = MODEL.predict_proba(np.array(rows))[:, 1]
    P = np.full((m, m), 0.5)
    for (i, j), p in zip(idx, probs):
        P[i][j] = p
    return P


def seed_order(n):
    order = [1]
    while len(order) < n:
        m = len(order) * 2 + 1
        order = [x for pair in ((o, m - o) for o in order) for x in pair]
    return order


ROUND_NAMES = {32: "1/16", 16: "1/8", 8: "1/4", 4: "1/2", 2: "Финал", 1: "Аварга"}


class BracketReq(BaseModel):
    ids: Optional[List[str]] = None
    size: int = 16
    sims: int = 20000


class PredictReq(BaseModel):
    w1: str
    w2: str


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.get("/api/players")
def players(q: str = "", limit: int = 40, active: int = 0):
    q = q.strip().lower()
    pool = ACTIVE if active else PLAYERS
    res = []
    for p in pool:
        name = full_name(p).lower()
        if q and q not in name and q not in (p.get("title_name") or "").lower():
            continue
        res.append({"id": p["id"], "name": full_name(p), "title_name": p.get("title_name") or "",
                    "elo": p.get("elo"), "matches": p["matches"], "winrate": p.get("winrate")})
        if len(res) >= limit:
            break
    return res


@app.get("/api/topseeds")
def topseeds(n: int = 16):
    """Elo-гоор эхний N идэвхтэй бөх (гараар засах анхны утга болгож)."""
    n = max(2, min(n, 32))
    return [{"id": p["id"], "name": full_name(p), "title_name": p.get("title_name") or "",
             "elo": p.get("elo")} for p in sorted(ACTIVE, key=lambda p: -(p.get("elo") or 0))[:n]]


@app.get("/api/profile/{wid}")
def profile(wid: str, limit: int = 300):
    p = PMAP.get(wid)
    if not p:
        raise HTTPException(404, "Бөх олдсонгүй")
    tnames = HIST.get("tnames", [])
    ent = HIST.get("hist", {}).get(wid, [])
    ent = sorted(ent, key=lambda e: e[0], reverse=True)[:limit]
    history = [{"date": e[0], "tournament": tnames[e[1]] if e[1] < len(tnames) else "",
                "rank": e[2], "opp_id": e[3], "opp": NAME.get(e[3], e[3]),
                "win": bool(e[4])} for e in ent]
    prof = dict(p)
    prof["name"] = full_name(p)
    prof["elo_rank"] = _rank.get(wid)
    prof["has_image"] = os.path.exists(os.path.join(IMAGES_DIR, p.get("img"))) if p.get("img") else False
    return {"player": prof, "history": history, "total": len(HIST.get("hist", {}).get(wid, []))}


_PLACEHOLDER = ('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="240">'
                '<rect width="200" height="240" fill="#e7eef5"/>'
                '<circle cx="100" cy="90" r="46" fill="#c2d4e5"/>'
                '<rect x="40" y="150" width="120" height="90" rx="30" fill="#c2d4e5"/>'
                '<text x="100" y="215" font-size="15" fill="#8199ad" text-anchor="middle">зураг алга</text></svg>')


@app.get("/api/photo/{wid}")
def photo(wid: str):
    p = PMAP.get(wid)
    fn = p.get("img") if p else None
    if fn:
        path = os.path.join(IMAGES_DIR, fn)
        if os.path.exists(path):
            return FileResponse(path)
    return Response(_PLACEHOLDER, media_type="image/svg+xml")


@app.post("/api/predict")
def predict(req: PredictReq):
    A, B = PMAP.get(req.w1), PMAP.get(req.w2)
    if not A or not B:
        raise HTTPException(404, "Бөх олдсонгүй")
    if req.w1 == req.w2:
        raise HTTPException(400, "Ижил бөх сонгосон байна")
    X, (h2hA, h2hB) = build_features(A, B)
    pA = float(MODEL.predict_proba(X)[0, 1])
    winner = A if pA >= 0.5 else B

    def info(p, prob):
        return {"id": p["id"], "name": full_name(p), "title_name": p.get("title_name"),
                "prob": round(prob, 4), "elo": p.get("elo"), "peak_elo": p.get("peak_elo"),
                "matches": p["matches"], "winrate": p.get("winrate"), "form": p.get("form"),
                "streak": p.get("streak"), "age": p.get("age"),
                "weight": p.get("weight"), "height": p.get("height")}

    return {"a": info(A, pA), "b": info(B, 1 - pA),
            "winner_id": winner["id"], "winner_name": full_name(winner),
            "h2h": {"a": h2hA, "b": h2hB}, "confidence": round(abs(pA - 0.5) * 2, 4)}


@app.post("/api/bracket")
def bracket(req: BracketReq):
    size = req.size if req.size in (2, 4, 8, 16, 32) else 16
    if req.ids:
        # Гараар оруулсан бол ОРУУЛСАН ДАРААЛЛААР нь торонд байрлуулна (1-2, 3-4, ...)
        ids = [i for i in req.ids if i in PMAP]
        if len(set(ids)) != len(ids):
            raise HTTPException(400, "Нэг бөх давхардсан байна")
        field = [PMAP[i] for i in ids]
        m = len(field)
        if m < 2 or (m & (m - 1)) != 0:
            raise HTTPException(400, "Бөхийн тоо 2, 4, 8, 16 эсвэл 32 байх ёстой")
        order = list(range(m))
    else:
        field = sorted(ACTIVE, key=lambda p: -(p.get("elo") or 0))[:size]
        m = len(field)
        order = [s - 1 for s in seed_order(m)]
    P = winprob_matrix(field)

    rounds = []
    cur = list(order)
    while len(cur) > 1:
        matchups, nxt = [], []
        for k in range(0, len(cur), 2):
            i, j = cur[k], cur[k + 1]
            pi = float(P[i][j]); win = i if pi >= 0.5 else j
            matchups.append({
                "a": {"id": field[i]["id"], "name": full_name(field[i]), "elo": field[i]["elo"]},
                "b": {"id": field[j]["id"], "name": full_name(field[j]), "elo": field[j]["elo"]},
                "pa": round(pi, 3), "winner": field[win]["id"]})
            nxt.append(win)
        rounds.append({"name": ROUND_NAMES.get(len(cur), f"1/{len(cur)//2}"), "matchups": matchups})
        cur = nxt
    champion = field[cur[0]]

    sims = max(1000, min(req.sims, 50000))
    rng = np.random.default_rng(42)
    champ = np.zeros(m); final = np.zeros(m)
    for _ in range(sims):
        c = list(order); fin = None
        while len(c) > 1:
            if len(c) == 2:
                fin = c[:]
            nx = []
            for k in range(0, len(c), 2):
                i, j = c[k], c[k + 1]
                nx.append(i if rng.random() < P[i][j] else j)
            c = nx
        champ[c[0]] += 1
        if fin:
            for x in fin:
                final[x] += 1
    odds = sorted(
        [{"id": field[i]["id"], "name": full_name(field[i]), "title_name": field[i].get("title_name"),
          "elo": field[i]["elo"], "champion": round(float(champ[i] / sims), 4),
          "finalist": round(float(final[i] / sims), 4)} for i in range(m)],
        key=lambda x: -x["champion"])

    return {"size": m, "sims": sims, "rounds": rounds,
            "champion": {"id": champion["id"], "name": full_name(champion),
                         "title_name": champion.get("title_name"), "elo": champion["elo"]},
            "odds": odds}


class GameReq(BaseModel):
    size: int = 32


@app.post("/api/game/new")
def game_new(req: GameReq):
    """Таамаглалын тоглоом: талбайг seed дарааллаар (0-1, 2-3 ... тулна) + магадлалын матриц.
    Frontend даваа бүрт хэрэглэгчийн сонголтыг авч, AI (argmax) ба санамсаргүй 'жинхэнэ'
    үр дүнтэй харьцуулж оноо тооцно."""
    size = req.size if req.size in (16, 32) else 32
    field = sorted(ACTIVE, key=lambda p: -(p.get("elo") or 0))[:size]
    order = [s - 1 for s in seed_order(size)]
    field = [field[i] for i in order]            # bracket дараалал
    P = winprob_matrix(field)
    return {
        "size": size,
        "field": [{"id": p["id"], "name": full_name(p), "title_name": p.get("title_name"),
                   "elo": p.get("elo"), "img": p.get("img")} for p in field],
        "P": [[round(float(P[i][j]), 4) for j in range(size)] for i in range(size)],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()
