"""Пять рисунков статьи PIERE 2026, ревизия v3. Все числа — из runs/ и table_data_v2.json."""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, "scripts")
import figstyle as st

st.apply()
OUT = "paper/figures"

TD = json.load(open("paper/tables/table_data_v2.json"))


def f1_binary(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def mdeberta_selection_v2():
    """gold-sel / silver-sel mDeBERTa на gold-test v2 из снимков epoch_cost."""
    gold = {}
    with open("data/gold/gold_test_v2.csv") as f:
        for row in csv.DictReader(f):
            gold[row["dialog_id"]] = int(row["is_scam"])
    import statistics as stat
    gsel, ssel = [], []
    for seed in (20260721, 20260722, 20260723):
        hist = json.load(open(f"runs/epoch_cost/data/mdeberta/history_seed{seed}.json"))
        sval, gdev = {}, {}
        for e in hist:
            ep = e.get("epoch")
            if ep is None:
                continue
            for k, v in e.items():
                if "silver_val" in k and k.endswith("f1"):
                    sval[round(ep)] = v
                if "gold_dev" in k and k.endswith("f1"):
                    gdev[round(ep)] = v
        snap = json.load(open(f"runs/epoch_cost/data/mdeberta/gold_test_per_epoch_seed{seed}.json"))
        y_true = [gold[i] for i in snap["dialog_ids"]]
        best_g = max(sorted(gdev), key=lambda ep: (gdev[ep], -ep))
        best_s = max(sorted(sval), key=lambda ep: (sval[ep], -ep))
        for target, best in ((gsel, best_g), (ssel, best_s)):
            idx = snap["epochs"].index(best)
            entry = snap["per_epoch"][idx]
            preds = entry.get("y_pred") if isinstance(entry, dict) else entry
            target.append(f1_binary(y_true, preds))
    return (stat.mean(gsel), stat.stdev(gsel)), (stat.mean(ssel), stat.stdev(ssel))


# ---------------------------------------------------------------- Fig. 1: pipeline
def fig1():
    fig, ax = plt.subplots(figsize=(st.FULL_WIDTH_IN, st.FULL_WIDTH_IN / 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    lw = 0.8

    def box(x, y, w, h, text, ec, fc="white", dashed=False, fs=st.FONTSIZE_ANNOT,
            tc="black", bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
            linewidth=lw, edgecolor=ec, facecolor=fc,
            linestyle=(0, (4, 2)) if dashed else "solid"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
                linespacing=1.25)

    def arrow(x0, y0, x1, y1, color, ls="solid"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=7, linewidth=1.0,
                     color=color, linestyle=ls, shrinkA=0, shrinkB=0))

    # input
    box(1, 9, 13.5, 15, "Corpus\n2000 dialogues\nru / en / zh\nsilver train 1392",
        st.NEUTRAL)
    # model families container
    box(19, 3.2, 20, 26.6, "", st.NEUTRAL, dashed=True)
    ax.text(29, 32.2, "model families", ha="center", fontsize=st.FONTSIZE_ANNOT,
            color=st.NEUTRAL, style="italic")
    box(20.6, 20.5, 16.8, 5.4, "frequency-based", st.NEUTRAL)
    box(20.6, 13.6, 16.8, 5.4, "frozen embeddings", st.NEUTRAL)
    box(20.6, 6.7, 16.8, 5.4, "fine-tuned encoder", st.NEUTRAL)
    # selection container
    box(45, 3.2, 24, 26.6, "", st.NEUTRAL, dashed=True)
    ax.text(57, 32.2, "checkpoint selection", ha="center",
            fontsize=st.FONTSIZE_ANNOT, color=st.NEUTRAL, style="italic")
    box(46.6, 17.4, 20.8, 8.2, "human-verified\ngold-dev, 100 dialogues",
        st.GOLD, tc=st.GOLD, bold=True)
    box(46.6, 6.4, 20.8, 8.2, "machine-labelled\nsilver-val, 205 dialogues",
        st.SILVER, tc=st.SILVER, bold=True)
    # test container
    box(75, 3.2, 24, 26.6, "", st.NEUTRAL, dashed=True)
    ax.text(87, 32.2, "human-verified test (303)", ha="center",
            fontsize=st.FONTSIZE_ANNOT, color=st.NEUTRAL, style="italic")
    box(76.6, 17.4, 20.8, 8.2, "F1 = 0.9344\n(detector quality)",
        st.GOLD, tc=st.GOLD, bold=True)
    box(76.6, 6.4, 20.8, 8.2, "F1 = 0.8757\n(the silver ceiling)",
        st.SILVER, tc=st.SILVER, bold=True)
    # arrows
    arrow(15.0, 16.5, 18.5, 16.5, st.NEUTRAL)
    arrow(39.5, 16.5, 44.5, 16.5, st.NEUTRAL)
    arrow(67.9, 21.5, 76.1, 21.5, st.GOLD)
    arrow(67.9, 10.5, 76.1, 10.5, st.SILVER, ls=(0, (4, 2)))

    st.save(fig, f"{OUT}/fig1_pipeline")
    plt.close(fig)


# ------------------------------------------------------- Fig. 2: divergence curves
def fig2():
    # v3.1: полная 5-эпоховая кривая из переигровки без ранней остановки
    # (runs/seeds_gold_dev_e5full/, эпохи 1-3 воспроизводят канонический прогон);
    # fallback — канонический 3-эпоховый прогон.
    import os
    src = ("runs/seeds_gold_dev_e5full/results.json"
           if os.path.exists("runs/seeds_gold_dev_e5full/results.json")
           else "runs/seeds_gold_dev/results.json")
    r = json.load(open(src))
    fig, ax = plt.subplots(figsize=(st.COL_WIDTH_IN, 2.2))
    styles = ["solid", (0, (4, 2)), (0, (1, 1.2))]
    for i, run in enumerate(r["runs"]):
        eps = [e["epoch"] for e in run["per_epoch"]]
        sv = [e["silver_val_f1"] for e in run["per_epoch"]]
        gd = [e["gold_dev_f1"] for e in run["per_epoch"]]
        ax.plot(eps, sv, color=st.SILVER, linestyle=styles[i], linewidth=1.1)
        ax.plot(eps, gd, color=st.GOLD, linestyle=styles[i], linewidth=1.1)
        # gold-selected checkpoint (best gold_dev, earliest on tie)
        bg = max(range(len(eps)), key=lambda j: (gd[j], -eps[j]))
        ax.plot(eps[bg], gd[bg], "o", color=st.GOLD, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.2, zorder=5)
        # silver-selected checkpoint (best silver_val, earliest on tie)
        bsp = max(range(len(eps)), key=lambda j: (sv[j], -eps[j]))
        ax.plot(eps[bsp], sv[bsp], "s", color=st.SILVER, markersize=5,
                markerfacecolor="white", markeredgewidth=1.2, zorder=5)
    n_ep = max(e["epoch"] for run in r["runs"] for e in run["per_epoch"])
    ax.text(1.04, 1.004, "silver-val F1 (machine labels)", color=st.SILVER,
            fontsize=st.FONTSIZE_ANNOT, va="bottom", ha="left")
    ax.text(n_ep - 0.03, 0.874, "gold-dev F1 (human labels)", color=st.GOLD,
            fontsize=st.FONTSIZE_ANNOT, va="top", ha="right")
    ax.set_xlabel("fine-tuning epoch")
    ax.set_ylabel("F1")
    ax.set_xticks(list(range(1, n_ep + 1)))
    ax.set_ylim(0.84, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    st.save(fig, f"{OUT}/fig2_divergence")
    plt.close(fig)


# ------------------------------------------------- Fig. 3: selection-gap dumbbells
def fig3():
    (mg, mgs), (ms, mss) = mdeberta_selection_v2()
    rows = [  # (label, gold_sel, silver_sel)
        ("XLM-R (fine-tuned)", TD["xlmr_gold_sel_v2"]["f1"][0],
         TD["xlmr_silver_sel_v2"]["f1"]),
        ("mDeBERTa-v3 (fine-tuned)", mg, ms),
        ("keywords (rule)", TD["keywords_v2"]["f1"],
         TD["keywords_silver_threshold"]["f1"]),
        ("e5 + LightGBM (frozen)", TD["layer2_sel"]["gold_dev"]["mean"],
         TD["layer2_sel"]["silver_val"]["mean"]),
        ("TF-IDF + LogReg (frozen)", TD["tfidf_v2_sel"]["gold_dev"],
         TD["tfidf_v2_sel"]["silver_val"]),
    ]
    ceiling = TD["xlmr_silver_sel_v2"]["f1"]
    fig, ax = plt.subplots(figsize=(st.COL_WIDTH_IN, 2.15))
    ys = list(range(len(rows)))[::-1]
    for y, (label, g, s) in zip(ys, rows):
        ax.plot([s, g], [y, y], color=st.NEUTRAL, linewidth=1.0, zorder=1)
        ax.plot(g, y, "o", color=st.GOLD, markersize=5.5, zorder=3)
        ax.plot(s, y, "s", color=st.SILVER, markersize=5, zorder=3)
        dg = "left" if g >= s else "right"
        off = 0.004 if g >= s else -0.004
        ax.text(g + off, y + 0.03, f"{g:.4f}", fontsize=st.FONTSIZE_ANNOT,
                color=st.GOLD, ha=dg, va="center")
        ax.text(s - off, y + 0.03, f"{s:.4f}", fontsize=st.FONTSIZE_ANNOT,
                color=st.SILVER, ha="right" if dg == "left" else "left", va="center")
    ax.axvline(ceiling, color=st.SILVER, linestyle=(0, (2, 2)), linewidth=0.8, zorder=0)
    ax.text(ceiling - 0.002, -0.62, "silver ceiling 0.8757", rotation=0,
            fontsize=st.FONTSIZE_ANNOT, color=st.SILVER, ha="center", va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=st.FONTSIZE_ANNOT)
    ax.set_xlabel("F1 on human-verified test (n = 303)")
    ax.set_xlim(0.855, 0.955)
    ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    # legend via colored text (works in BW through marker shapes)
    ax.plot([], [], "o", color=st.GOLD, label="selected on human labels")
    ax.plot([], [], "s", color=st.SILVER, label="selected on machine labels")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4,
              borderaxespad=0.1)
    st.save(fig, f"{OUT}/fig3_selection_gap")
    plt.close(fig)


# ------------------------------------------------------------ Fig. 4: forest plot
def fig4():
    s = json.load(open("runs/significance/results.json"))
    fig, ax = plt.subplots(figsize=(st.COL_WIDTH_IN, 2.15))
    labels_en = {
        ("xlmr", "keywords"): "XLM-R vs keywords",
        ("xlmr", "mdeberta"): "XLM-R vs mDeBERTa (ens.)",
        ("xlmr", "layer2"): "XLM-R vs e5+LightGBM",
        ("xlmr", "tfidf"): "XLM-R vs TF-IDF",
        ("keywords", "tfidf"): "keywords vs TF-IDF",
    }
    pairs = s["pairs"]
    ys = list(range(len(pairs)))[::-1]
    for y, p in zip(ys, pairs):
        sig = (not p["test_only"]["crosses_zero"]
               and not p["test_and_seed"]["crosses_zero"]
               and p["permutation"]["p_value"] < 0.05)
        col = st.ACCENT if sig else st.NEUTRAL
        lo1, hi1 = p["test_only"]["ci95"]
        lo2, hi2 = p["test_and_seed"]["ci95"]
        ax.plot([lo1, hi1], [y + 0.13, y + 0.13], color=col, linewidth=1.6,
                solid_capstyle="butt")
        ax.plot([lo2, hi2], [y - 0.13, y - 0.13], color=col, linewidth=0.7,
                solid_capstyle="butt")
        ax.plot(p["observed_diff"], y + 0.13, "D", color=col, markersize=3.6)
        ptxt = f"p = {p['permutation']['p_value']:.4f}"
        ax.text(0.113, y, ptxt + ("  *" if sig else "  n.s."),
                fontsize=st.FONTSIZE_ANNOT, color=col, va="center", ha="left")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([labels_en[(p["a"], p["b"])] for p in pairs],
                       fontsize=st.FONTSIZE_ANNOT)
    ax.set_xlabel("F1 difference (positive favours first model)")
    ax.set_xlim(-0.06, 0.165)
    ax.spines[["top", "right"]].set_visible(False)
    st.save(fig, f"{OUT}/fig4_significance")
    plt.close(fig)


# -------------------------------------------------------- Fig. 5: transfer heatmap
def fig5():
    c = json.load(open("runs/crosslingual/results.json"))
    x = c["models"]["xlmr"]
    mat = x["by_selection"]["dev_source"]["matrix"]
    cfgs = x["configs"]
    order = ["single_en", "single_ru", "single_zh",
             "loo_en+ru", "loo_en+zh", "loo_ru+zh",
             "loo_en+ru_n464", "loo_en+zh_n464", "loo_ru+zh_n464",
             "all3", "all3_n464"]
    row_names = {
        "single_en": "en only", "single_ru": "ru only", "single_zh": "zh only",
        "loo_en+ru": "en+ru", "loo_en+zh": "en+zh", "loo_ru+zh": "ru+zh",
        "loo_en+ru_n464": "en+ru", "loo_en+zh_n464": "en+zh",
        "loo_ru+zh_n464": "ru+zh", "all3": "all three",
        "all3_n464": "all three"}
    langs = ["en", "ru", "zh"]
    zero_shot = {
        "single_en": {"ru", "zh"}, "single_ru": {"en", "zh"},
        "single_zh": {"en", "ru"},
        "loo_en+ru": {"zh"}, "loo_en+zh": {"ru"}, "loo_ru+zh": {"en"},
        "loo_en+ru_n464": {"zh"}, "loo_en+zh_n464": {"ru"},
        "loo_ru+zh_n464": {"en"}, "all3": set(), "all3_n464": set()}
    vals = [[mat[cfg][lg]["f1"]["median"] for lg in langs] for cfg in order]
    fig, ax = plt.subplots(figsize=(st.COL_WIDTH_IN, 2.95))
    cmap = plt.get_cmap("viridis")
    vmin, vmax = 0.72, 1.0
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i, cfg in enumerate(order):
        n = cfgs[cfg]["composition"]["n_total"]
        for j, lg in enumerate(langs):
            v = vals[i][j]
            r, g, b, _ = cmap((v - vmin) / (vmax - vmin))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            tc = "black" if lum > 0.55 else "white"
            star = "*" if lg in zero_shot[cfg] else ""
            ax.text(j, i, f"{v:.3f}{star}", ha="center", va="center",
                    fontsize=st.FONTSIZE_ANNOT, color=tc)
    # frame the en->zh anomaly cell (row single_en, col zh)
    ax.add_patch(Rectangle((2 - 0.5, 0 - 0.5), 1, 1, fill=False,
                           edgecolor=st.SILVER, linewidth=1.6))
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"test {lg}" for lg in langs])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{row_names[c_]} (n={cfgs[c_]['composition']['n_total']})"
                        for c_ in order], fontsize=st.FONTSIZE_ANNOT)
    ax.set_xlabel("test language")
    ax.set_ylabel("training languages (train size)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("median F1", fontsize=st.FONTSIZE_ANNOT)
    cb.ax.tick_params(labelsize=st.FONTSIZE_ANNOT)
    st.save(fig, f"{OUT}/fig5_transfer")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures written to", OUT)
