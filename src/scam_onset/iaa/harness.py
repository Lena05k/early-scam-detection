#!/usr/bin/env python3
"""IAA-харнесс для калибровки трёхъязычного скам-корпуса с онсетом."""
import csv, json, argparse, itertools
from collections import defaultdict, Counter

SUBS = ["sub_phish", "sub_money", "sub_other"]

def _onset(x):
    x = (x or "").strip()
    return int(x) if x != "" else None

def load_gold(path):
    g = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["item_id"])
            g[i] = dict(T=int(r["T"]), round=r.get("round", ""),
                        is_scam=int(r["is_scam"]),
                        subs=tuple(int(r[s]) for s in SUBS),
                        onset=_onset(r["onset"]),
                        hn=int(r.get("hard_negative", 0) or 0))
    return g

def load_annotations(path):
    a = defaultdict(dict)
    skipped = Counter()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["item_id"]); who = r["annotator"]
            if (r["is_scam"] or "").strip() == "":  # незаполненный шаблон — item ещё не размечен
                skipped[who] += 1
                continue
            a[i][who] = dict(is_scam=int(r["is_scam"]),
                             subs=tuple(int(r[s] or 0) for s in SUBS),
                             onset=_onset(r["onset"]))
    if skipped:
        det = ", ".join(f"{w}: {n}" for w, n in sorted(skipped.items()))
        print(f"⚠ пропущены незаполненные строки (пустой is_scam) — {det}")
    return a

def cohen_kappa(a, b):
    n = len(a)
    if n == 0: return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c]/n) * (cb[c]/n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

def fleiss_kappa(rows_counts, k_categories):
    N = len(rows_counts)
    if N == 0: return float("nan")
    n = sum(rows_counts[0].values())
    if n < 2: return float("nan")
    P_i = []
    for rc in rows_counts:
        s = sum(v*(v-1) for v in rc.values())
        P_i.append(s / (n*(n-1)))
    Pbar = sum(P_i)/N
    col = Counter()
    for rc in rows_counts:
        for c, v in rc.items(): col[c] += v
    total = N*n
    Pe = sum((col[c]/total)**2 for c in k_categories)
    return 1.0 if Pe == 1 else (Pbar - Pe)/(1 - Pe)

def kappa_for(items, extract, categories=None):
    raters = sorted({w for it in items for w in items[it]})
    if len(raters) == 2:
        ra, rb = raters
        pairs = [(extract(items[it][ra]), extract(items[it][rb]))
                 for it in items if ra in items[it] and rb in items[it]]
        if not pairs: return float("nan")
        A, B = zip(*pairs)
        return cohen_kappa(list(A), list(B))
    cats = categories or sorted({extract(items[it][w]) for it in items for w in items[it]})
    rows = []
    for it in items:
        rc = Counter()
        for w in items[it]:
            rc[extract(items[it][w])] += 1
        for c in cats: rc.setdefault(c, 0)
        rows.append(rc)
    return fleiss_kappa(rows, cats)

def krippendorff_alpha_interval(units):
    coin = defaultdict(float)
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2: continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coin[(vals[i], vals[j])] += w
    if not coin: return float("nan")
    n_v = defaultdict(float)
    for (a, b), c in coin.items(): n_v[a] += c
    n = sum(n_v.values())
    if n < 2: return float("nan")
    Do = sum(c * (a - b)**2 for (a, b), c in coin.items()) / n
    De = 0.0
    vals = list(n_v)
    for a in vals:
        for b in vals:
            De += n_v[a]*n_v[b]*(a - b)**2
    De /= (n*(n-1))
    return float("nan") if De == 0 else 1 - Do/De

def median_lower(vs):
    s = sorted(vs); k = len(s)
    if k == 0: return None
    return s[k//2] if k % 2 else s[k//2 - 1]

def run(gold_path, annot_path, save_json=None):
    gold = load_gold(gold_path)
    ann = load_annotations(annot_path)
    raters = sorted({w for it in ann for w in ann[it]})
    print("="*64)
    print(f"IAA-отчёт | аннотаторов: {len(raters)} ({', '.join(raters)}) | items: {len(gold)}")
    print("="*64)
    k_scam = kappa_for(ann, lambda r: r["is_scam"], categories=[0, 1])
    stat = "Cohen" if len(raters) == 2 else "Fleiss"
    print(f"\n[1] is_scam  {stat}'s kappa = {k_scam:.3f}")
    print("\n[2] subtypes (per-label kappa):")
    ks = []
    for idx, s in enumerate(SUBS):
        k = kappa_for(ann, lambda r, i=idx: r["subs"][i], categories=[0, 1])
        ks.append(k); print(f"    {s:11s} kappa = {k:.3f}")
    ks_ok = [x for x in ks if x == x]
    macro = sum(ks_ok)/len(ks_ok) if ks_ok else float("nan")
    print(f"    macro-kappa = {macro:.3f}" if ks_ok else "    macro-kappa = n/a")
    co_scam = [it for it in ann
               if all(ann[it][w]["is_scam"] == 1 and ann[it][w]["onset"] is not None
                      for w in ann[it])]
    units = [[ann[it][w]["onset"] for w in ann[it]] for it in co_scam]
    alpha = krippendorff_alpha_interval(units)
    print(f"\n[3] onset t*  Krippendorff alpha (interval) = {alpha:.3f}   (со-скам items: {len(co_scam)})")
    diffs = []
    for it in co_scam:
        ons = [ann[it][w]["onset"] for w in ann[it]]
        for a, b in itertools.combinations(ons, 2):
            diffs.append(abs(a - b))
    mae = agree1 = float("nan")
    if diffs:
        mae = sum(diffs)/len(diffs)
        agree1 = sum(1 for d in diffs if d <= 1)/len(diffs)
        print(f"    pairwise onset-MAE = {mae:.2f} реплик | agreement@pm1 = {agree1:.3f}")
    print("\n[4] воспроизведение gold:")
    for w in raters:
        sc_ok = sc_tot = on_ok = on_pm1 = on_tot = sub_ok = sub_tot = 0
        for it in gold:
            if w not in ann[it]: continue
            r = ann[it][w]; g = gold[it]
            sc_tot += 1; sc_ok += int(r["is_scam"] == g["is_scam"])
            sub_tot += 1; sub_ok += int(r["subs"] == g["subs"])
            if g["is_scam"] == 1 and g["onset"] is not None:
                on_tot += 1
                if r["onset"] is not None:
                    on_ok += int(r["onset"] == g["onset"])
                    on_pm1 += int(abs(r["onset"] - g["onset"]) <= 1)
        print(f"    {w}: is_scam acc={sc_ok}/{sc_tot} | subtype-set acc={sub_ok}/{sub_tot} | "
              f"onset exact={on_ok}/{on_tot} | onset+-1={on_pm1}/{on_tot}")
    print("\n[5] агрегированный t*_gold (медиана):")
    agg = {}
    for it in gold:
        votes = [ann[it][w] for w in ann[it]]
        n_scam = sum(v["is_scam"] for v in votes)
        if n_scam * 2 < len(votes):
            agg[it] = None
        else:
            ons = [v["onset"] for v in votes if v["is_scam"] == 1 and v["onset"] is not None]
            agg[it] = median_lower(ons)
    for cs in range(0, len(sorted(agg)), 8):
        ids = sorted(agg)[cs:cs+8]
        print("    " + "  ".join(f"Pr.{it}:{agg[it] if agg[it] is not None else 'None'}" for it in ids))
    print("\n[6] лог расхождений:")
    n_dis = 0
    for it in sorted(gold):
        recs = ann[it]; ws = sorted(recs)
        tags = []
        if len({recs[w]["is_scam"] for w in ws}) > 1: tags.append("is_scam")
        if len({recs[w]["subs"] for w in ws}) > 1: tags.append("subtypes")
        ons = {w: recs[w]["onset"] for w in ws}
        if len(set(ons.values())) > 1: tags.append("onset")
        vs_gold = [w for w in ws if (recs[w]["is_scam"], recs[w]["subs"], recs[w]["onset"])
                   != (gold[it]["is_scam"], gold[it]["subs"], gold[it]["onset"])]
        if tags or vs_gold:
            n_dis += 1; g = gold[it]
            detail = " ".join(f"{w}:{ons[w]}" for w in ws)
            print(f"    Pr.{it:>2} [{g['round']}] {','.join(tags) or 'vs-gold':<16} "
                  f"gold(scam={g['is_scam']},t*={g['onset']}) | onset {detail}"
                  + (f" | off-gold: {','.join(vs_gold)}" if vs_gold else ""))
    print(f"\n    итого items с расхождением/off-gold: {n_dis}/{len(gold)}")
    print("\n[7] критерии заморозки v1->v1.1:")
    def mark(ok): return "PASS" if ok else "FAIL"
    a1 = k_scam >= 0.80
    a2 = (alpha == alpha) and alpha >= 0.67
    a3 = bool(diffs) and agree1 >= 0.90
    print(f"    is_scam kappa >= 0.80        : {mark(a1)} ({k_scam:.3f})")
    print(f"    onset alpha >= 0.67          : {mark(a2)} ({alpha:.3f})")
    print(f"    onset agreement@pm1 >= 0.90  : {mark(a3)}")
    print(f"    все R2 (Pr.19-24) разрешены  : РУЧНАЯ проверка в протоколе")
    print("="*64)
    if save_json:
        res = dict(is_scam_kappa=k_scam, subtype_macro=macro, sub_other=ks[2],
                   onset_alpha=alpha, onset_agree_pm1=agree1, onset_mae=mae,
                   n_coscam=len(co_scam), n_items=len(gold), n_annotators=len(raters))
        res = {k: (None if isinstance(v, float) and v != v else v) for k, v in res.items()}
        with open(save_json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"результаты сохранены: {save_json}")

def _cli():
    ap = argparse.ArgumentParser(description="IAA-калибровка")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--save-json", default=None, help="куда сохранить результаты (JSON)")
    a = ap.parse_args()
    run(a.gold, a.annotations, save_json=a.save_json)

if __name__ == "__main__":
    _cli()
