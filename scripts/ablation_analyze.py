#!/usr/bin/env python3
"""Разбор ablation диалогового контекста: снимки Kaggle → runs/ablation_context/results.json.

Обучение здесь не идёт — всё считается из пер-эпохных снимков вероятностей реплик
(`runs/ablation_context/data/*.json`, ноутбук `notebooks/ablation_context_kaggle.ipynb`)
и из готовых пер-диалоговых вероятностей эталонной диалоговой модели
(`runs/earliness_v2/prefix_probs_by_seed.json`, последний префикс = полный диалог;
воспроизводит 0.9437/0.9243/0.9351, среднее 0.9344 ± 0.0097).

Что считается:
  1. отбор эпохи и правила агрегации — ТОЛЬКО по gold-dev 100;
  2. метрики на gold-test v2 (303) для всех конфигураций;
  3. сравнение трёх правил агрегации и развёртка по доле k;
  4. смещение агрегации по длине диалога (ошибки и ложные тревоги против T);
  5. где расходятся диалоговая и turn-модель (2×2, состав зон);
  6. пореплико́вое поведение turn-модели по позициям (до/после онсета);
  7. значимость: парный бутстрап 10000 + перестановочный тест.

  python scripts/ablation_analyze.py
"""
import argparse
import csv
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ablation_context import SEEDS, aggregate, f1_np, rule_grid  # noqa: E402
from crosslingual import load_all  # noqa: E402
from significance import paired_bootstrap, permutation_test  # noqa: E402
from train_binary import compute_metrics_from  # noqa: E402

N_BOOT, N_PERM, BOOT_SEED = 10000, 10000, 20260811
ONSET_BINS = [("1", lambda t: t == 1), ("2", lambda t: t == 2),
              ("3–4", lambda t: 3 <= t <= 4), ("5+", lambda t: t >= 5)]


# ─────────────────────────── загрузка снимков ───────────────────────────
def load_runs(data_dir):
    """{(конфигурация, сид): запись прогона} из снимков Kaggle."""
    runs = {}
    for f in sorted(glob.glob(str(pathlib.Path(data_dir) / "*.json"))):
        name = pathlib.Path(f).stem
        # retrain_* и weights_* — переобучение ради сохранения весов, это те же
        # конфигурации; если их подхватить, они затрут снимки основных прогонов
        if name.startswith(("retrain_", "weights_")):
            continue
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        runs[(d["config"], d["seed"])] = d
    return runs


def group_probs(probs, dialog_ids):
    out = {}
    for did, p in zip(dialog_ids, probs):
        out.setdefault(did, []).append(float(p))
    return out


def select(run, raw, order_dev, gold_dev):
    """Эпоха × правило агрегации по gold-dev. Тай-брейк: ранняя эпоха, порядок сетки."""
    y = np.array([gold_dev[d] for d in order_dev])
    best = {"f1": -1.0}
    per_epoch = []
    for e, probs in enumerate(run["snapshots"]["gold_dev"], 1):
        g = group_probs(probs, raw["turn_dialog_ids"]["gold_dev"])
        row = {"epoch": e}
        for rule, param in rule_grid():
            f = f1_np(y, np.array([aggregate(g[d], rule, param) for d in order_dev]))
            if f > best["f1"]:
                best = {"f1": f, "epoch": e, "rule": rule, "param": param}
            if rule == "any":
                row["any_f1"] = f
        per_epoch.append(row)
    # лучшее по каждому правилу отдельно — для графика сравнения правил
    by_rule = {}
    for rule in ("any", "frac", "maxprob"):
        b = {"f1": -1.0}
        for e, probs in enumerate(run["snapshots"]["gold_dev"], 1):
            g = group_probs(probs, raw["turn_dialog_ids"]["gold_dev"])
            for r2, param in rule_grid():
                if r2 != rule:
                    continue
                f = f1_np(y, np.array([aggregate(g[d], r2, param) for d in order_dev]))
                if f > b["f1"]:
                    b = {"f1": f, "epoch": e, "rule": r2, "param": param}
        by_rule[rule] = b
    return best, by_rule, per_epoch


def test_preds(run, raw, sel, order_test):
    """Предсказания на gold-test при выбранных эпохе и правиле + вероятности реплик."""
    probs = run["snapshots"]["gold_test"][sel["epoch"] - 1]
    g = group_probs(probs, raw["turn_dialog_ids"]["gold_test"])
    pred = np.array([aggregate(g[d], sel["rule"], sel["param"]) for d in order_test])
    return pred, g


def metrics_block(y, pred, langs, silver, ids):
    m = compute_metrics_from(y, pred)
    m["confusion"] = [[int(np.sum((y == a) & (pred == b))) for b in (0, 1)]
                      for a in (0, 1)]
    m["per_language"] = {lg: compute_metrics_from(y[langs == lg], pred[langs == lg])["f1"]
                         for lg in ("en", "ru", "zh")}
    m["silver_agreement"] = float(np.mean(pred == silver))
    return m


def mstd(vals):
    vals = [float(v) for v in vals]
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals}


def agg_metrics(per_seed):
    keys = ["accuracy", "precision", "recall", "f1", "cohen_kappa", "silver_agreement"]
    out = {k: mstd([s[k] for s in per_seed]) for k in keys}
    out["per_language"] = {lg: mstd([s["per_language"][lg] for s in per_seed])
                           for lg in ("en", "ru", "zh")}
    out["confusion_mean"] = np.mean([s["confusion"] for s in per_seed],
                                    axis=0).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="runs/ablation_context")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    root = pathlib.Path(args.dir)

    raw = json.loads((root / "results_raw.json").read_text(encoding="utf-8"))
    runs = load_runs(root / "data")
    cfg_names = [c["name"] for c in raw["configs"]]
    print(f"конфигураций {len(cfg_names)}, прогонов {len(runs)}")

    data = load_all()
    test_recs = data["test"]
    order_test = raw["dialog_order"]["gold_test"]
    order_dev = raw["dialog_order"]["gold_dev"]
    gold_dev = raw["dialog_gold"]["gold_dev"]
    by_id = {r["id"]: r for r in test_recs}
    assert sorted(order_test) == sorted(r["id"] for r in test_recs), "порядок теста разошёлся"

    y = np.array([by_id[d]["gold_is_scam"] for d in order_test])
    langs = np.array([by_id[d]["lang"] for d in order_test])
    lengths = np.array([len(by_id[d]["turns"]) for d in order_test])
    hardneg = np.array([int(by_id[d].get("hard_negative", 0)) for d in order_test])
    # silver-метки теста: исходная колонка is_scam корпуса (та, что выбрасывается
    # при джойне с gold) — нужна только чтобы измерить «совпадение с silver»
    silver_map = {}
    with open("data/corpus/corpus_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") == "test":
                silver_map[rec["id"]] = int(rec["is_scam"])
    silver = np.array([silver_map[d] for d in order_test])

    gold_onset = {}
    with open("data/gold/gold_test_v2.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            o = (row.get("onset") or "").strip()
            gold_onset[row["dialog_id"]] = int(o) if o else None

    # ── эталон: диалоговая модель (последний префикс = полный диалог) ──
    pp = json.loads(pathlib.Path(
        "runs/earliness_v2/prefix_probs_by_seed.json").read_text(encoding="utf-8"))
    dlg_preds = {}
    for s in SEEDS:
        pr = pp[str(s)]
        for d in order_test:
            assert len(pr[d]) == len(by_id[d]["turns"]), f"префиксов не T: {d}"
        dlg_preds[s] = np.array([int(pr[d][-1] >= 0.5) for d in order_test])
    dlg_per_seed = [metrics_block(y, dlg_preds[s], langs, silver, order_test)
                    for s in SEEDS]
    configs = {"dialogue": {"n_train": len(data["train"]),
                            "unit": "диалог", "selection": "эталон, отбор по gold-dev",
                            "per_seed": dlg_per_seed, **agg_metrics(dlg_per_seed)}}
    print(f"\nэталон dialogue: F1 {configs['dialogue']['f1']['mean']:.4f} ± "
          f"{configs['dialogue']['f1']['std']:.4f}")

    # ── turn-конфигурации ──
    preds_by_cfg = {"dialogue": {s: dlg_preds[s] for s in SEEDS}}
    epoch_mismatch = []
    turn_probs_by_cfg = {}
    sel_by_cfg = {}
    for name in cfg_names:
        per_seed, sels, rules, per_epoch_all, tprobs = [], [], [], {}, {}
        preds_by_cfg[name] = {}
        for s in SEEDS:
            run = runs[(name, s)]
            sel, by_rule, per_epoch = select(run, raw, order_dev, gold_dev)
            # снимки округлены до 6 знаков, метрика в ядре считалась на полной
            # точности — расхождение возможно только на границе; крупное расхождение
            # означает рассинхрон снимков и метрики, и это уже поломка
            if sel["epoch"] != run["best_epoch"]:
                gap = abs(sel["f1"] - run["dev_bestagg_f1_by_epoch"][run["best_epoch"] - 1])
                msg = (f"{name} сид {s}: эпоха постфактум {sel['epoch']} против "
                       f"{run['best_epoch']} в ядре, разрыв по dev F1 {gap:.5f}")
                assert gap <= 0.005, msg
                print(f"  ⚠ {msg} (в пределах допуска, беру постфактум)")
                epoch_mismatch.append(msg)
            pred, g = test_preds(run, raw, sel, order_test)
            preds_by_cfg[name][s] = pred
            tprobs[s] = g
            per_seed.append(metrics_block(y, pred, langs, silver, order_test))
            sels.append(sel)
            rules.append(by_rule)
            per_epoch_all[s] = per_epoch
        turn_probs_by_cfg[name] = tprobs
        sel_by_cfg[name] = sels
        comp = next(c["composition"] for c in raw["configs"] if c["name"] == name)
        m = agg_metrics(per_seed)
        # вырожденность агрегации: «почти всё scam» или «почти всё benign»
        degen = [bool((s["recall"] >= 0.95 and s["precision"] <= 0.70)
                      or s["recall"] <= 0.10) for s in per_seed]
        configs[name] = {
            "n_train": comp["n"], "n_train_dialogs": comp["n_dialogs"],
            "unit": "реплика", "scheme": name.split("_")[1],
            "size_matched": name.endswith("n1392"),
            "selection": [{"seed": s, **sel} for s, sel in zip(SEEDS, sels)],
            "selection_by_rule": [{"seed": s, **{k: v for k, v in r.items()}}
                                  for s, r in zip(SEEDS, rules)],
            "dev_per_epoch": per_epoch_all,
            "epochs_run": [runs[(name, s)]["epochs"] for s in SEEDS],
            "minutes": [runs[(name, s)]["minutes"] for s in SEEDS],
            "degenerate_seeds": degen,
            "per_seed": per_seed, **m,
        }
        print(f"  {name:<22} n={comp['n']:>6}  F1 {m['f1']['mean']:.4f} ± "
              f"{m['f1']['std']:.4f}  правила "
              f"{[(x['rule'], x['param']) for x in sels]}  эпохи "
              f"{[x['epoch'] for x in sels]}"
              + ("  ⚠ вырожденные сиды" if any(degen) else ""))

    # лучшая turn-конфигурация — ТОЛЬКО по gold-dev
    dev_mean = {n: float(np.mean([s["f1"] for s in sel_by_cfg[n]])) for n in cfg_names}
    best_turn = max(dev_mean, key=lambda k: (dev_mean[k], k))
    if best_turn != raw["best_config"]:
        # веса в models/ сохранены для конфигурации, выбранной ЯДРОМ, — если выбор
        # разошёлся, об этом надо знать явно, а не молча
        print(f"  ⚠ лучшая конфигурация постфактум {best_turn} против "
              f"{raw['best_config']} в ядре — веса сохранены для второй")
        epoch_mismatch.append(f"best_config: постфактум {best_turn}, "
                              f"в ядре {raw['best_config']}")
    print(f"\nлучшая turn-конфигурация по gold-dev: {best_turn} "
          f"(dev F1 {dev_mean[best_turn]:.4f})")

    # ── сравнение правил агрегации: dev и test ──
    agg_cmp = {}
    for name in cfg_names:
        rows = {}
        for rule in ("any", "frac", "maxprob"):
            dev_f1, test_f1 = [], []
            for k, s in enumerate(SEEDS):
                br = configs[name]["selection_by_rule"][k][rule]
                dev_f1.append(br["f1"])
                pred, _ = test_preds(runs[(name, s)], raw, br, order_test)
                test_f1.append(f1_np(y, pred))
            rows[rule] = {"gold_dev": mstd(dev_f1), "gold_test": mstd(test_f1),
                          "params": [configs[name]["selection_by_rule"][k][rule]["param"]
                                     for k in range(len(SEEDS))]}
        agg_cmp[name] = rows

    # развёртка по доле k (при выбранной эпохе): dev и test
    frac_sweep = {}
    for name in cfg_names:
        ks = [round(x, 3) for x in np.arange(0.0, 1.001, 0.02)]
        sweep = {"k": ks, "gold_dev": [], "gold_test": [], "k_selected": []}
        for k_i, s in enumerate(SEEDS):
            run = runs[(name, s)]
            ep = sel_by_cfg[name][k_i]["epoch"]
            gd = group_probs(run["snapshots"]["gold_dev"][ep - 1],
                             raw["turn_dialog_ids"]["gold_dev"])
            gt = group_probs(run["snapshots"]["gold_test"][ep - 1],
                             raw["turn_dialog_ids"]["gold_test"])
            ydev = np.array([gold_dev[d] for d in order_dev])
            sweep["gold_dev"].append(
                [f1_np(ydev, np.array([aggregate(gd[d], "frac", k) for d in order_dev]))
                 for k in ks])
            sweep["gold_test"].append(
                [f1_np(y, np.array([aggregate(gt[d], "frac", k) for d in order_test]))
                 for k in ks])
            sweep["k_selected"].append(
                configs[name]["selection_by_rule"][k_i]["frac"]["param"])
        frac_sweep[name] = sweep

    # ── смещение по длине ──
    def by_length(pred_by_seed):
        uniq = sorted(set(lengths.tolist()))
        err, fa, n_all, n_ben = [], [], [], []
        for T in uniq:
            mask = lengths == T
            mb = mask & (y == 0)
            err.append(mstd([float(np.mean(pred_by_seed[s][mask] != y[mask]))
                             for s in SEEDS]))
            n_all.append(int(mask.sum()))
            n_ben.append(int(mb.sum()))
            fa.append(mstd([float(np.mean(pred_by_seed[s][mb] == 1)) for s in SEEDS])
                      if mb.sum() else None)
        return {"lengths": uniq, "n": n_all, "n_benign": n_ben,
                "error_rate": err, "false_alarm_rate": fa}

    length_block = {}
    for name in ["dialogue", best_turn]:
        p = preds_by_cfg[name]
        def corr(a, b):
            """Пирсон; None, если один из рядов константен (корреляция не определена)."""
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if a.std() == 0 or b.std() == 0:
                return None
            return float(np.corrcoef(a, b)[0, 1])

        cs = [corr(lengths, (p[s] != y).astype(float)) for s in SEEDS]
        fs = [corr(lengths[y == 0], (p[s][y == 0] == 1).astype(float)) for s in SEEDS]
        cs = [c for c in cs if c is not None] or [float("nan")]
        fs = [c for c in fs if c is not None] or [float("nan")]
        length_block[name] = {
            **by_length(p),
            "corr_error_vs_length": mstd(cs),
            "corr_false_alarm_vs_length": mstd(fs),
            "len_correct": mstd([float(np.mean(lengths[p[s] == y])) for s in SEEDS]),
            "len_wrong": mstd([float(np.mean(lengths[p[s] != y])) for s in SEEDS]),
            "len_correct_all": [lengths[p[s] == y].tolist() for s in SEEDS],
            "len_wrong_all": [lengths[p[s] != y].tolist() for s in SEEDS],
        }

    # ── где расходятся модели (голосование по сидам) ──
    def vote(name):
        return (np.mean([preds_by_cfg[name][s] for s in SEEDS], axis=0) >= 0.5).astype(int)

    onset_of = np.array([gold_onset.get(d) or 0 for d in order_test])

    def zones_vs(turn_name):
        vd, vt = vote("dialogue"), vote(turn_name)
        zones = {"оба правы": (vd == y) & (vt == y),
                 "прав только dialogue": (vd == y) & (vt != y),
                 "прав только turn": (vd != y) & (vt == y),
                 "ошиблись оба": (vd != y) & (vt != y)}
        out = {}
        for zname, mask in zones.items():
            bins = {}
            for bn, pred in ONSET_BINS:
                in_bin = np.array([bool(o) and pred(o) for o in onset_of])
                bins[bn] = int(np.sum(mask & (y == 1) & in_bin))
            out[zname] = {"n": int(mask.sum()),
                          "hard_negative": int(np.sum(mask & (hardneg == 1))),
                          "gold_scam": int(np.sum(mask & (y == 1))),
                          "by_onset_bin": bins,
                          "mean_length": float(lengths[mask].mean()) if mask.sum() else None}
        return out

    agreement_by_cfg = {n: zones_vs(n) for n in cfg_names}
    agreement = agreement_by_cfg[best_turn]

    # ── пореплико́вое поведение turn-моделей (считается для КАЖДОЙ конфигурации:
    # у naive-схемы «правильная» метка реплики по онсету — не то, чему её учили,
    # поэтому доля срабатываний там описательная, а не оценка качества) ──
    def turn_level_block(tp):
        pos_rows, pre_post = {}, {}
        max_pos = int(lengths.max())
        for pos in range(1, max_pos + 1):
            acc, n = [], 0
            for s in SEEDS:
                hit, tot = 0, 0
                for d in order_test:
                    probs = tp[s][d]
                    if len(probs) < pos:
                        continue
                    o = gold_onset.get(d)
                    lab = int(by_id[d]["gold_is_scam"] == 1 and o and pos >= o)
                    hit += int((probs[pos - 1] >= 0.5) == bool(lab))
                    tot += 1
                acc.append(hit / tot if tot else np.nan)
                n = tot
            pos_rows[pos] = {"n": n, "accuracy": mstd(acc)}
        for tag, cond in (("до онсета", lambda i, o: i < o), ("на онсете и после", lambda i, o: i >= o)):
            acc, share = [], []
            for s in SEEDS:
                hit = tot = fired = 0
                for d in order_test:
                    o = gold_onset.get(d)
                    if not (by_id[d]["gold_is_scam"] == 1 and o):
                        continue
                    for i, p in enumerate(tp[s][d], 1):
                        if not cond(i, o):
                            continue
                        lab = 1 if i >= o else 0
                        hit += int((p >= 0.5) == bool(lab))
                        tot += 1
                        fired += int(p >= 0.5)
                acc.append(hit / tot if tot else np.nan)
                share.append(fired / tot if tot else np.nan)
            pre_post[tag] = {"accuracy": mstd(acc), "share_predicted_scam": mstd(share),
                             "n_turns": tot}
        benign_share = []
        for s in SEEDS:
            fired = tot = 0
            for d in order_test:
                if by_id[d]["gold_is_scam"] == 1:
                    continue
                for p in tp[s][d]:
                    fired += int(p >= 0.5)
                    tot += 1
            benign_share.append(fired / tot)
        pre_post["реплики benign-диалогов"] = {"share_predicted_scam": mstd(benign_share),
                                               "n_turns": tot}

        # профили вероятностей для 20 случайных скам-диалогов (график 10)
        rng = np.random.default_rng(BOOT_SEED)
        scam_ids = [d for d in order_test if by_id[d]["gold_is_scam"] == 1 and gold_onset.get(d)]
        sample = sorted(rng.permutation(len(scam_ids))[:20])
        profiles = [{"id": scam_ids[i], "onset": gold_onset[scam_ids[i]],
                     "probs": [float(np.mean([tp[s][scam_ids[i]][j] for s in SEEDS]))
                               for j in range(len(tp[SEEDS[0]][scam_ids[i]]))]}
                    for i in sample]
        return {"by_position": pos_rows, "pre_post_onset": pre_post,
                "profiles": profiles}

    turn_level_by_cfg = {n: turn_level_block(turn_probs_by_cfg[n]) for n in cfg_names}

    # ── значимость ──
    print(f"\nпарный бутстрап ({args.n_boot}) и перестановочный тест ({args.n_perm}):")
    rng = np.random.default_rng(BOOT_SEED)
    sig = []
    for a, b in [("dialogue", best_turn)] + [("dialogue", n) for n in cfg_names
                                             if n != best_turn]:
        pa = {s: preds_by_cfg[a][s] for s in SEEDS}
        pb = {s: preds_by_cfg[b][s] for s in SEEDS}
        pa["_vote"], pb["_vote"] = vote(a), vote(b)
        row = {"a": a, "b": b, "f1_a": f1_np(y, pa["_vote"]), "f1_b": f1_np(y, pb["_vote"])}
        row["observed_diff"] = row["f1_a"] - row["f1_b"]
        for tag, mix in (("test_only", False), ("test_and_seed", True)):
            d = paired_bootstrap(y, pa, pb, rng, args.n_boot, mix)
            lo, hi = np.percentile(d, [2.5, 97.5])
            row[tag] = {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                        "p_a_better": float((d > 0).mean()),
                        "crosses_zero": bool(lo <= 0 <= hi)}
        obs, p = permutation_test(y, pa["_vote"], pb["_vote"], rng, args.n_perm)
        row["permutation"] = {"observed_abs_diff": float(obs), "p_value": float(p)}
        sig.append(row)
        print(f"  {a} против {b}: ΔF1 {row['observed_diff']:+.4f} | тест "
              f"[{row['test_only']['ci95'][0]:+.4f}, {row['test_only']['ci95'][1]:+.4f}] | "
              f"тест+сиды [{row['test_and_seed']['ci95'][0]:+.4f}, "
              f"{row['test_and_seed']['ci95'][1]:+.4f}] | p={p:.4f} | "
              f"{'НЕ значимо' if row['test_and_seed']['crosses_zero'] else 'значимо'}")

    # ── лесенка и согласие с silver из уже посчитанных слоёв ──
    l2 = json.loads(pathlib.Path("runs/layer2/results.json").read_text(encoding="utf-8"))
    ladder = [r for r in l2["ladder"]]
    for name in cfg_names:
        ladder.append({"model": f"ablation: {name}", "f1": configs[name]["f1"]["mean"],
                       "std": configs[name]["f1"]["std"], "layer": "ablation", "gold": "v2"})
    silver_by_model = dict(l2["silver_agreement_by_model"])
    try:
        l4 = json.loads(pathlib.Path(
            "runs/layer4/results_control.json").read_text(encoding="utf-8"))
        silver_by_model["контроль: deepseek-chat"] = float(
            l4["summary"]["silver_agreement"]["mean"])
    except Exception as e:                      # noqa: BLE001 — источник необязателен
        print(f"⚠ согласие DeepSeek не подхватилось: {e}")
    for name in cfg_names:
        silver_by_model[f"ablation: {name}"] = configs[name]["silver_agreement"]["mean"]

    payload = {
        "task": "ablation диалогового контекста: диалог против одиночной реплики",
        "gold": "data/gold/gold_test_v2.csv", "n_test": int(len(y)),
        "seeds": SEEDS, "hyperparams": raw["hyperparams"],
        "dialogue_reference": {
            "source": "runs/earliness_v2/prefix_probs_by_seed.json (последний префикс)",
            "note": "веса models/binary_xlmr_seed*, воспроизводят 0.9344 ± 0.0097"},
        "best_turn_config": best_turn, "dev_mean_by_config": dev_mean,
        "configs": configs,
        "aggregation_comparison": agg_cmp,
        "frac_sweep": frac_sweep,
        "length_bias": length_block,
        "agreement_2x2": agreement,
        "turn_level": turn_level_by_cfg[best_turn],
        "turn_level_by_config": turn_level_by_cfg,
        "agreement_2x2_by_config": agreement_by_cfg,
        "significance": {"n_boot": args.n_boot, "n_perm": args.n_perm,
                         "boot_seed": BOOT_SEED, "pairs": sig},
        "ladder": ladder, "silver_agreement_by_model": silver_by_model,
        "retrain": raw.get("retrain", []),
        "selection_mismatch": epoch_mismatch,
        "gpu": raw.get("gpu"), "minutes_total": float(
            sum(sum(configs[n]["minutes"]) for n in cfg_names)),
    }
    (root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nсохранено: {root / 'results.json'}")


if __name__ == "__main__":
    main()
