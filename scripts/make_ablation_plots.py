#!/usr/bin/env python3
"""Графики ablation диалогового контекста. Числа только из runs/ablation_context/results.json.

  python scripts/make_ablation_plots.py
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

DPI = 150
FIG = (10, 6)
FIG2 = (14, 6)
C_DLG = "#4C72B0"
C_TURN = "#C44E52"
C_TURN2 = "#DD8452"
SEED_COLORS = ["#4C72B0", "#C44E52", "#55A868"]

RU = {"dialogue": "диалог (эталон)",
      "turn_naive_full": "реплика, метка диалога\n(полный набор)",
      "turn_naive_n1392": "реплика, метка диалога\n(1392)",
      "turn_onset_full": "реплика, метка по онсету\n(полный набор)",
      "turn_onset_n1392": "реплика, метка по онсету\n(1392)"}
SHORT = {"dialogue": "диалог", "turn_naive_full": "реплика/метка диалога, полный",
         "turn_naive_n1392": "реплика/метка диалога, 1392",
         "turn_onset_full": "реплика/метка онсета, полный",
         "turn_onset_n1392": "реплика/метка онсета, 1392"}
RULE_RU = {"any": "хотя бы одна", "frac": "доля k из T", "maxprob": "максимум вероятности"}


def order(p):
    """Порядок конфигураций на всех графиках: эталон, затем turn-варианты."""
    names = ["dialogue"] + [n for n in ("turn_naive_full", "turn_naive_n1392",
                                        "turn_onset_full", "turn_onset_n1392")
                            if n in p["configs"]]
    return names


def bar_color(n):
    """Цвет = схема разметки реплик, штриховка = вариант с контролем размера."""
    if n == "dialogue":
        return C_DLG
    return C_TURN if "naive" in n else C_TURN2


def bar_hatch(n):
    return "//" if n.endswith("n1392") else None


# ─────────────────────────────── основные ───────────────────────────────
def context_ablation(p, path):
    names = order(p)
    mu = [p["configs"][n]["f1"]["mean"] for n in names]
    sd = [p["configs"][n]["f1"]["std"] for n in names]
    ntr = [p["configs"][n]["n_train"] for n in names]
    fig, ax = plt.subplots(figsize=FIG)
    x = np.arange(len(names))
    bars = ax.bar(x, mu, 0.62, yerr=sd, capsize=5,
                  color=[bar_color(n) for n in names], edgecolor="white")
    for b, n in zip(bars, names):
        b.set_hatch(bar_hatch(n))
    ref = p["configs"]["dialogue"]["f1"]["mean"]
    ax.axhline(ref, color=C_DLG, ls="--", lw=1.4)
    ax.text((len(names) - 1) / 2, ref + 0.022, f"уровень диалоговой модели {ref:.4f}",
            ha="center", fontsize=9, color=C_DLG,
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5})
    for xi, m, s, n in zip(x, mu, sd, ntr):
        ax.text(xi, m + s + 0.016, f"{m:.4f}", ha="center", fontsize=9.5, weight="bold")
        ax.text(xi, 0.03, f"n={n}", ha="center", fontsize=9, color="#333")
    ax.set_xticks(x, [RU[n] for n in names], fontsize=9)
    ax.set_ylabel("F1 на gold-test v2 (n=303)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Ablation диалогового контекста: единица обучения и предсказания\n"
                 "усы = std по 3 сидам, подпись снизу — число обучающих примеров",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def full_metrics(p, path):
    names = order(p)
    keys = [("f1", "F1"), ("precision", "precision"), ("recall", "recall"),
            ("cohen_kappa", "Cohen κ")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (k, title) in zip(axes.ravel(), keys):
        mu = [p["configs"][n][k]["mean"] for n in names]
        sd = [p["configs"][n][k]["std"] for n in names]
        x = np.arange(len(names))
        ax.bar(x, mu, 0.62, yerr=sd, capsize=4,
               color=[bar_color(n) for n in names], edgecolor="white")
        ax.axhline(p["configs"]["dialogue"][k]["mean"], color=C_DLG, ls="--", lw=1.2)
        for xi, m, s in zip(x, mu, sd):
            ax.text(xi, m + s + 0.015, f"{m:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, [RU[n].replace("\n", " ") for n in names], fontsize=7.5,
                      rotation=12)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.1)
    fig.suptitle("Чем платит turn-модель: F1, precision, recall, κ на gold-test v2\n"
                 "пунктир — уровень диалоговой модели, усы = std по 3 сидам",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def significance(p, path):
    pairs = p["significance"]["pairs"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ylab = []
    for k, row in enumerate(pairs):
        for tag, col, off in (("test_only", "#8172B3", +0.16),
                              ("test_and_seed", "#937860", -0.16)):
            d = row[tag]
            yy = len(pairs) - 1 - k + off
            ax.plot([d["ci95"][0], d["ci95"][1]], [yy, yy], color=col, lw=2.6,
                    solid_capstyle="round",
                    label=("только тест" if tag == "test_only" else "тест + сиды")
                    if k == 0 else None)
            ax.plot([d["mean_diff"]], [yy], "o", color=col, ms=6)
        ylab.append(f"диалог − {SHORT[row['b']]}\n"
                    f"ΔF1 {row['observed_diff']:+.4f}, p={row['permutation']['p_value']:.4f}")
    ax.axvline(0, color="#444", lw=1.5)
    ax.set_ylim(-0.6, len(pairs) - 0.2)
    ax.set_yticks(range(len(pairs))[::-1], ylab, fontsize=9)
    ax.set_xlabel("разница F1 (диалог − реплика), 95% ДИ парного бутстрапа")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.set_title("Значимость выигрыша диалогового контекста\n"
                 f"парный бутстрап {p['significance']['n_boot']} реплик; "
                 "ДИ, пересекающий ноль, = разница не показана", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ─────────────────────────────── агрегация ───────────────────────────────
def aggregation_comparison(p, path):
    best = p["best_turn_config"]
    rules = ["any", "frac", "maxprob"]
    names = [n for n in order(p) if n != "dialogue"]
    fig, axes = plt.subplots(1, 2, figsize=FIG2, sharey=True)
    for ax, field, title in ((axes[0], "gold_dev", "Отбор: gold-dev (n=100)"),
                             (axes[1], "gold_test", "Замер: gold-test v2 (n=303)")):
        x = np.arange(len(names))
        w = 0.26
        for j, rule in enumerate(rules):
            mu = [p["aggregation_comparison"][n][rule][field]["mean"] for n in names]
            sd = [p["aggregation_comparison"][n][rule][field]["std"] for n in names]
            ax.bar(x + (j - 1) * w, mu, w, yerr=sd, capsize=3,
                   color=SEED_COLORS[j], edgecolor="white", label=RULE_RU[rule])
            for xi, m in zip(x + (j - 1) * w, mu):
                ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x, [RU[n] for n in names], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("F1")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Три способа собрать реплики в диалог. Правило выбирается ТОЛЬКО по "
                 f"gold-dev\nлучшая конфигурация по dev: {RU[best].replace(chr(10), ' ')}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def threshold_sweep(p, path):
    best = p["best_turn_config"]
    sw = p["frac_sweep"][best]
    ks = sw["k"]
    fig, ax = plt.subplots(figsize=FIG)
    for j, s in enumerate(p["seeds"]):
        ax.plot(ks, sw["gold_test"][j], color=SEED_COLORS[j], lw=1.8, label=f"сид {s}")
        ax.plot(ks, sw["gold_dev"][j], color=SEED_COLORS[j], lw=1.1, ls=":", alpha=0.7)
        ax.axvline(sw["k_selected"][j], color=SEED_COLORS[j], ls="--", lw=1.0, alpha=0.8)
    ax.axhline(p["configs"]["dialogue"]["f1"]["mean"], color="#333", ls="-.", lw=1.3,
               label="диалоговая модель")
    ax.set_xlabel("k — требуемая доля реплик, предсказанных как scam")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"Развёртка порога доли: {RU[best].replace(chr(10), ' ')}\n"
                 "сплошная — gold-test, пунктир-точки — gold-dev, "
                 "вертикаль — k, выбранное по dev", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ─────────────────────────── смещение по длине ───────────────────────────
def error_by_length(p, path):
    fig, ax = plt.subplots(figsize=FIG)
    for name, col in (("dialogue", C_DLG), (p["best_turn_config"], C_TURN)):
        b = p["length_bias"][name]
        x = b["lengths"]
        mu = np.array([r["mean"] for r in b["error_rate"]])
        sd = np.array([r["std"] for r in b["error_rate"]])
        ax.plot(x, mu, "-o", color=col, lw=2, ms=5,
                label=RU[name].replace("\n", " "))
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.18)
    b = p["length_bias"]["dialogue"]
    for xi, n in zip(b["lengths"], b["n"]):
        ax.text(xi, -0.028, f"{n}", ha="center", fontsize=7.5, color="#555")
    ax.set_xlabel("число реплик в диалоге (снизу — сколько диалогов)")
    ax.set_ylabel("доля ошибок")
    ax.set_ylim(-0.04, None)
    ax.legend(fontsize=9)
    cd = p["length_bias"]["dialogue"]["corr_error_vs_length"]["mean"]
    ct = p["length_bias"][p["best_turn_config"]]["corr_error_vs_length"]["mean"]
    ax.set_title("Ошибки против длины диалога, полоса = std по 3 сидам\n"
                 f"корреляция ошибки с длиной: диалог {cd:+.3f}, реплика {ct:+.3f}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def false_alarm_by_length(p, path):
    fig, ax = plt.subplots(figsize=FIG)
    for name, col in (("dialogue", C_DLG), (p["best_turn_config"], C_TURN)):
        b = p["length_bias"][name]
        xs = [x for x, r in zip(b["lengths"], b["false_alarm_rate"]) if r]
        mu = np.array([r["mean"] for r in b["false_alarm_rate"] if r])
        sd = np.array([r["std"] for r in b["false_alarm_rate"] if r])
        ax.plot(xs, mu, "-o", color=col, lw=2, ms=5, label=RU[name].replace("\n", " "))
        ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.18)
    b = p["length_bias"]["dialogue"]
    for xi, n in zip(b["lengths"], b["n_benign"]):
        if n:
            ax.text(xi, -0.028, f"{n}", ha="center", fontsize=7.5, color="#555")
    cd = p["length_bias"]["dialogue"]["corr_false_alarm_vs_length"]["mean"]
    ct = p["length_bias"][p["best_turn_config"]]["corr_false_alarm_vs_length"]["mean"]
    ax.set_xlabel("число реплик в benign-диалоге (снизу — сколько диалогов)")
    ax.set_ylabel("доля ложных тревог")
    ax.set_ylim(-0.04, None)
    ax.legend(fontsize=9)
    ax.set_title("Главная проверка агрегации: растёт ли риск ложной тревоги с длиной\n"
                 f"корреляция с длиной: диалог {cd:+.3f}, реплика {ct:+.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def length_distribution(p, path):
    names = ["dialogue", p["best_turn_config"]]
    fig, axes = plt.subplots(1, 2, figsize=FIG2, sharey=True)
    allb = p["length_bias"]
    lo = min(min(allb[n]["lengths"]) for n in names)
    hi = max(max(allb[n]["lengths"]) for n in names)
    bins = np.arange(lo - 0.5, hi + 1.5, 1)
    for ax, name in zip(axes, names):
        ok = np.concatenate([np.array(v) for v in allb[name]["len_correct_all"]])
        bad = np.concatenate([np.array(v) for v in allb[name]["len_wrong_all"]])
        ax.hist([ok, bad], bins=bins, density=True, color=["#55A868", "#C44E52"],
                label=[f"верно (сред. {allb[name]['len_correct']['mean']:.2f})",
                       f"ошибка (сред. {allb[name]['len_wrong']['mean']:.2f})"])
        ax.set_title(RU[name].replace("\n", " "), fontsize=11)
        ax.set_xlabel("число реплик")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("плотность (все три сида вместе)")
    fig.suptitle("Распределение длин: верно и неверно классифицированные диалоги",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ─────────────────────── где расходятся модели ───────────────────────
def agreement(p, path):
    z = p["agreement_2x2"]
    keys = ["оба правы", "прав только dialogue", "прав только turn", "ошиблись оба"]
    fig, axes = plt.subplots(1, 2, figsize=FIG2,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    m = np.array([[z["оба правы"]["n"], z["прав только dialogue"]["n"]],
                  [z["прав только turn"]["n"], z["ошиблись оба"]["n"]]])
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=20,
                    color="white" if m[i, j] > m.max() * 0.55 else "#222")
    ax.set_xticks([0, 1], ["turn прав", "turn ошибся"], fontsize=10)
    ax.set_yticks([0, 1], ["dialogue прав", "dialogue ошибся"], fontsize=10)
    ax.set_title("Голосование по 3 сидам, gold-test v2 (303)", fontsize=11)
    ax.grid(False)

    ax = axes[1]
    x = np.arange(len(keys))
    hn = [z[k]["hard_negative"] for k in keys]
    bins = list(p["agreement_2x2"]["оба правы"]["by_onset_bin"])
    bottom = np.zeros(len(keys))
    for j, bn in enumerate(bins):
        v = np.array([z[k]["by_onset_bin"][bn] for k in keys], dtype=float)
        ax.bar(x - 0.2, v, 0.36, bottom=bottom, label=f"скам, t*={bn}",
               color=plt.cm.Blues(0.35 + 0.16 * j), edgecolor="white")
        bottom += v
    ax.bar(x + 0.2, hn, 0.36, color="#937860", edgecolor="white",
           label="hard_negative корпуса")
    ax.set_xticks(x, [k.replace(" только ", "\nтолько ") for k in keys], fontsize=8.5)
    ax.set_ylabel("диалогов")
    ax.legend(fontsize=8.5)
    ax.set_title("Состав зон: бины истинного t* и hard-negatives", fontsize=11)
    fig.suptitle("Где расходятся диалоговая и turn-модель "
                 f"({SHORT[p['best_turn_config']]}, выбрана по gold-dev)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def onset_cfg(p):
    """Конфигурация, обученная РАЗЛИЧАТЬ реплики (метка по онсету, контроль размера).

    Именно к ней осмысленно применять пореплико́вые графики: naive-схема училась
    ставить метку диалога на каждую реплику и различать их внутри диалога не обязана.
    """
    for n in ("turn_onset_n1392", "turn_onset_full"):
        if n in p["configs"]:
            return n
    return p["best_turn_config"]


def prob_profiles(p, path):
    cfg = onset_cfg(p)
    prof = p["turn_level_by_config"][cfg]["profiles"]
    T = max(len(x["probs"]) for x in prof)
    m = np.full((len(prof), T), np.nan)
    for k, x in enumerate(prof):
        m[k, :len(x["probs"])] = x["probs"]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(m, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    for k, x in enumerate(prof):
        ax.plot([x["onset"] - 1.5, x["onset"] - 1.5], [k - 0.5, k + 0.5], color="#111",
                lw=2.4)
    ax.set_xticks(range(T), [str(i + 1) for i in range(T)], fontsize=8)
    ax.set_yticks(range(len(prof)), [x["id"].replace("corpus_", "") for x in prof],
                  fontsize=7)
    ax.set_xlabel("номер реплики")
    fig.colorbar(im, ax=ax, label="P(scam) для отдельной реплики, среднее по 3 сидам")
    ax.grid(False)
    ax.set_title(f"Turn-модель по репликам ({SHORT[cfg]}): "
                 "20 случайных скам-диалогов теста\n"
                 "чёрная черта — истинный онсет t*", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def turn_by_position(p, path):
    """Обе схемы разметки сразу: обучали ли различать реплики — видно по контрасту."""
    cfgs = [n for n in ("turn_onset_n1392", "turn_naive_n1392") if n in p["configs"]]
    cols = {"turn_onset_n1392": C_TURN2, "turn_naive_n1392": C_TURN}
    fig, axes = plt.subplots(1, 2, figsize=FIG2,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for cfg in cfgs:
        pos = p["turn_level_by_config"][cfg]["by_position"]
        xs = sorted(int(k) for k in pos)
        mu = np.array([pos[str(k)]["accuracy"]["mean"] for k in xs])
        sd = np.array([pos[str(k)]["accuracy"]["std"] for k in xs])
        ax.plot(xs, mu, "-o", color=cols[cfg], lw=2, ms=5, label=SHORT[cfg])
        ax.fill_between(xs, mu - sd, mu + sd, color=cols[cfg], alpha=0.18)
    pos = p["turn_level_by_config"][cfgs[0]]["by_position"]
    for k in sorted(int(x) for x in pos):
        ax.text(k, 0.02, f"{pos[str(k)]['n']}", ha="center", fontsize=7, color="#555")
    ax.set_xlabel("позиция реплики (снизу — сколько реплик на этой позиции)")
    ax.set_ylabel("точность на уровне реплики")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Точность реплики против её позиции\n"
                 "(эталон реплики — метка по золотому онсету)", fontsize=11)

    ax = axes[1]
    keys = ["до онсета", "на онсете и после", "реплики benign-диалогов"]
    x = np.arange(len(keys))
    w = 0.38
    for j, cfg in enumerate(cfgs):
        pp = p["turn_level_by_config"][cfg]["pre_post_onset"]
        v = [pp[k]["share_predicted_scam"]["mean"] for k in keys]
        e = [pp[k]["share_predicted_scam"]["std"] for k in keys]
        ax.bar(x + (j - 0.5) * w, v, w, yerr=e, capsize=3, color=cols[cfg],
               edgecolor="white", label=SHORT[cfg])
        for xi, val, err in zip(x + (j - 0.5) * w, v, e):
            ax.text(xi, val + err + 0.02, f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [k.replace(" ", "\n", 1) for k in keys], fontsize=8.5)
    ax.set_ylabel("доля реплик, предсказанных как scam")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Различает ли модель реплики внутри диалога", fontsize=11)
    fig.suptitle("Пореплико́вое поведение turn-моделей (обе схемы, контроль размера)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ─────────────────────── связь с остальной работой ───────────────────────
def silver_agreement(p, path):
    d = p["silver_agreement_by_model"]
    items = sorted(d.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cols = ["#C44E52" if k.startswith("ablation") else
            "#937860" if k.startswith("контроль") else "#4C72B0" for k, _ in items]
    ax.barh([k.replace("ablation: ", "ablation: ") for k, _ in items],
            [v for _, v in items], color=cols, edgecolor="white")
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=9)
    ax.axvline(0.8614, color="#333", ls="--", lw=1.3,
               label="сама silver-разметка против человека (0.8614)")
    ax.set_xlim(0.7, 1.03)
    ax.set_xlabel("доля совпадений с silver-метками на gold-test")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Кто воспроизводит черновую разметку DeepSeek\n"
                 "красным — turn-конфигурации этого ablation", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def ladder(p, path):
    rows = sorted(p["ladder"], key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(11, 7))
    cols = ["#C44E52" if r["layer"] == "ablation" else
            {"слой 0": "#937860", "слой 1": "#DD8452", "обученные": "#DD8452",
             "слой 2": "#55A868", "файнтюн": "#4C72B0"}.get(r["layer"], "#8172B3")
            for r in rows]
    ax.barh([r["model"] for r in rows], [r["f1"] for r in rows], color=cols,
            edgecolor="white",
            xerr=[r.get("std", 0) for r in rows], capsize=3)
    for i, r in enumerate(rows):
        ax.text(r["f1"] + r.get("std", 0) + 0.014, i, f"{r['f1']:.4f}", va="center",
                fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 на gold-test (303 диалога)")
    ax.set_title("Лесенка качества с turn-конфигурациями (красным)\n"
                 "⚠ TF-IDF и mDeBERTa — числа на gold v1, остальные на v2", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def per_language(p, path):
    names = order(p)
    langs = ["en", "ru", "zh"]
    fig, ax = plt.subplots(figsize=FIG)
    x = np.arange(len(langs))
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        mu = [p["configs"][n]["per_language"][lg]["mean"] for lg in langs]
        sd = [p["configs"][n]["per_language"][lg]["std"] for lg in langs]
        off = (j - (len(names) - 1) / 2) * w
        ax.bar(x + off, mu, w, yerr=sd, capsize=3, color=bar_color(n),
               alpha=1.0 if n == "dialogue" else 0.55 + 0.15 * (j % 2),
               edgecolor="white", label=RU[n].replace("\n", " "))
        for xi, m in zip(x + off, mu):
            ax.text(xi, m + 0.015, f"{m:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x, langs, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1")
    ax.legend(fontsize=8, ncol=2, loc="lower center")
    ax.set_title("F1 по языкам: диалог против реплики (gold-test v2, ~101 диалог на язык)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="runs/ablation_context")
    args = ap.parse_args()
    root = pathlib.Path(args.dir)
    p = json.loads((root / "results.json").read_text(encoding="utf-8"))
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (context_ablation, "context_ablation.png"),
        (full_metrics, "ablation_full_metrics.png"),
        (significance, "ablation_significance.png"),
        (aggregation_comparison, "aggregation_comparison.png"),
        (threshold_sweep, "aggregation_threshold_sweep.png"),
        (error_by_length, "error_by_length.png"),
        (false_alarm_by_length, "false_alarm_by_length.png"),
        (length_distribution, "length_distribution_by_error.png"),
        (agreement, "dialogue_vs_turn_agreement.png"),
        (prob_profiles, "turn_prob_profiles.png"),
        (turn_by_position, "turn_level_by_position.png"),
        (silver_agreement, "silver_agreement_with_turn.png"),
        (ladder, "ladder_with_ablation.png"),
        (per_language, "per_language_ablation.png"),
    ]
    for fn, name in jobs:
        fn(p, out / name)
        print(f"  {name}")
    print(f"\n{len(jobs)} графиков в {out} (dpi={DPI})")


if __name__ == "__main__":
    main()
