"""Сборка piere2026_paper_en.docx / piere2026_report_ru.docx из шаблона PIERE 2026.

Собирает из КОПИИ оригинального шаблона: стили, колонки и автонумерация
(заголовки I./A., ссылки [n], подписи Fig. n. / TABLE n.) — шаблонные.
Формулы — OMML, номер справа. Рис. 1 — плавающая таблица на всю полосу.
"""
import re
import shutil
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import qn, nsmap
from docx.shared import Emu, Inches, Pt

TEMPLATE = "docs/PIERE2026__full_paper_template.docx"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = nsmap["w"]

# ----------------------------------------------------------------- OMML helpers
def mr(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<m:r><m:t xml:space="preserve">{t}</m:t></m:r>'

def msub(base, sub):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"

def msup(base, sup):
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"

def msubsup(base, sub, sup):
    return (f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub>"
            f"<m:sup>{sup}</m:sup></m:sSubSup>")

def mhat(base):
    return ('<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr>'
            f"<m:e>{base}</m:e></m:acc>")

def mlimlow(base, low):
    return f"<m:limLow><m:e>{base}</m:e><m:lim>{low}</m:lim></m:limLow>"

def mnary(sub, sup, body):
    return ('<m:nary><m:naryPr><m:chr m:val="∑"/>'
            '<m:limLoc m:val="undOvr"/></m:naryPr>'
            f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>")

def omath(inner):
    return parse_xml(f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}">{inner}</m:oMath>')

FORMULAS = {
    1: (mr("C=(") + msub(mr("u"), mr("1")) + mr(", …, ")
        + msub(mr("u"), mr("T")) + mr("),   ") + msub(mr("f"), mr("θ"))
        + mr(" : C→[0, 1],   ") + mhat(mr("y")) + mr("=1[")
        + msub(mr("f"), mr("θ")) + mr("(C)≥τ]")),
    2: (mhat(mr("θ")) + mr("(S)=")
        + mlimlow(mr("argmax"), mr("θ∈{") + msub(mr("θ"), mr("1"))
                  + mr(",…,") + msub(mr("θ"), mr("E")) + mr("}"))
        + mr("F1(θ; ") + msubsup(mr("D"), mr("val"), mr("S"))
        + mr("),  S∈{gold, silver}")),
    3: (msub(mr("Δ"), mr("sel")) + mr("=F1(") + mhat(mr("θ"))
        + mr("(gold); ") + msubsup(mr("D"), mr("test"), mr("gold"))
        + mr(") − F1(") + mhat(mr("θ")) + mr("(silver); ")
        + msubsup(mr("D"), mr("test"), mr("gold")) + mr(")")),
    4: (msub(mr("Δ"), mr("zs")) + mr("(l)=F1(l | ")
        + msub(mr("L"), mr("src")) + mr(", n) − F1(l | {l}, n),  l∉")
        + msub(mr("L"), mr("src"))),
    5: (mr("δ=") + msub(mr("F1"), mr("A")) + mr(" − ")
        + msub(mr("F1"), mr("B")) + mr(",  p=(1/B)")
        + mnary(mr("b=1"), mr("B"),
                mr("1[|") + msup(mr("δ"), mr("(b)")) + mr("|≥|δ|]"))),
    6: (msub(mr("v"), mr("C")) + mr("=(1/T)")
        + mnary(mr("t=1"), mr("T"), mr("e(") + msub(mr("u"), mr("t")) + mr(")"))),
}

# --------------------------------------------------------------- md text runs
SUBSCRIPT_RE = re.compile(r"(?<=[A-Za-zθΔ])_([A-Za-z0-9]+(?:\^[A-Za-z0-9]+)?)")

def add_runs(par, text, italic_all=False):
    """**bold** / *italic* / X_sub / X_sub^sup из markdown-текста."""
    text = text.replace("[REPO_URL].", " REPOURL.").replace("[REPO_URL]", " REPOURL ")
    parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for part in parts:
        bold = italic = False
        if part.startswith("**") and part.endswith("**"):
            bold, part = True, part[2:-2]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            italic, part = True, part[1:-1]
        pos = 0
        for m in SUBSCRIPT_RE.finditer(part):
            if m.start() > pos:
                r = par.add_run(part[pos:m.start()])
                r.bold, r.italic = bold, italic or italic_all
            token = m.group(1)
            if "^" in token:
                sub_t, sup_t = token.split("^", 1)
            else:
                sub_t, sup_t = token, None
            r = par.add_run(sub_t)
            r.font.subscript = True
            r.bold, r.italic = bold, italic or italic_all
            if sup_t:
                r = par.add_run(sup_t)
                r.font.superscript = True
                r.bold, r.italic = bold, italic or italic_all
            pos = m.end()
        if pos < len(part):
            r = par.add_run(part[pos:].replace(" REPOURL ", " [REPO_URL] ")
                            .replace(" REPOURL.", " [REPO_URL]."))
            r.bold, r.italic = bold, italic or italic_all

# ------------------------------------------------------------------ md parser
def parse_md(path):
    lines = open(path, encoding="utf8").read().split("\n")
    title = lines[0].lstrip("# ").strip()
    text = "\n".join(lines)
    abstract = re.search(r"## (?:Abstract|Аннотация)\n\n(.*?)\n\n\*\*", text,
                         re.S).group(1).strip()
    kw = re.search(r"\*\*(?:Keywords|Ключевые слова):\*\* (.*)", text).group(1).strip()
    body_m = re.search(r"\n(## I\..*?)\n## (?:References|Список источников)",
                       text, re.S)
    body_lines = body_m.group(1).split("\n")
    refs = re.findall(r"\n\[(\d+)\] (.+)", text)
    blocks = []
    for ln in body_lines:
        ln = ln.rstrip()
        if not ln:
            continue
        if ln.startswith("## "):
            h = re.sub(r"^[IVX]+\.\s*", "", ln[3:])
            blocks.append(("h1", h))
        elif ln.startswith("### "):
            h = re.sub(r"^[A-Z]\.\s*", "", ln[4:])
            blocks.append(("h2", h))
        else:
            fm = re.match(r".*\((\d)\)\s*$", ln)
            if fm and "=" in ln and "**" not in ln and len(ln) < 200:
                blocks.append(("formula", int(fm.group(1))))
            else:
                blocks.append(("p", ln))
    return title, abstract, kw, blocks, refs

def parse_md_table(path):
    lines = [l for l in open(path, encoding="utf8").read().split("\n") if l.strip()]
    title = lines[0].split(". ", 1)[1] if ". " in lines[0] else lines[0]
    rows = []
    for l in lines[1:]:
        if set(l.replace("|", "").strip()) <= {"-"}:
            continue
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows.append(cells)
    return title, rows

# --------------------------------------------------------------- docx helpers
def clear_paragraph(p):
    for r in list(p._p):
        if r.tag != qn("w:pPr"):
            p._p.remove(r)

def add_equation_par(doc, num, style="equation"):
    p = doc.add_paragraph(style=style)
    pPr = p._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:tabs")):
        pPr.remove(old)
    tabs = parse_xml(
        f'<w:tabs xmlns:w="{W_NS}">'
        '<w:tab w:val="center" w:pos="2380"/>'
        '<w:tab w:val="right" w:pos="4890"/></w:tabs>')
    pPr.append(tabs)
    r = p.add_run(); r._r.append(parse_xml(f'<w:tab xmlns:w="{W_NS}"/>'))
    p._p.append(omath(FORMULAS[num]))
    r2 = p.add_run(); r2._r.append(parse_xml(f'<w:tab xmlns:w="{W_NS}"/>'))
    p.add_run(f"({num})")
    return p

def make_table(doc, rows, width_in=3.38, header_style="table col head",
               body_style="table copy"):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders xmlns:w="{W_NS}">'
        + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                  for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders>")
    tblPr.append(borders)
    ncols = len(rows[0])
    first_w = int(width_in * 1440 * (1.5 / (ncols + 0.5)))
    rest_w = int((width_in * 1440 - first_w) / max(1, ncols - 1))
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Emu((first_w if j == 0 else rest_w) * 635)
            p = cell.paragraphs[0]
            p.style = doc.styles[header_style if i == 0 else body_style]
            add_runs(p, cell_text.replace("+-", "±"))
    return t

def add_figure(doc, img, width_in, caption, cap_style="figure caption"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.add_run().add_picture(img, width=Inches(width_in))
    cp = doc.add_paragraph(style=cap_style)
    add_runs(cp, caption)
    return p, cp

def add_float_fig1(doc, anchor_par, img, caption):
    """Плавающая таблица на всю полосу (7.16in) вверху страницы якоря."""
    t = doc.add_table(rows=2, cols=1)
    tblPr = t._tbl.tblPr
    tblPr.append(parse_xml(
        f'<w:tblpPr xmlns:w="{W_NS}" w:leftFromText="144" w:rightFromText="144" '
        'w:topFromText="144" w:bottomFromText="230" w:vertAnchor="page" '
        'w:horzAnchor="margin" w:tblpXSpec="center" w:tblpY="1085"/>'))
    tblPr.append(parse_xml(
        f'<w:tblW xmlns:w="{W_NS}" w:w="{int(7.01 * 1440)}" w:type="dxa"/>'))
    c0 = t.cell(0, 0); p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.add_run().add_picture(img, width=Inches(7.01))
    c1 = t.cell(1, 0); p1 = c1.paragraphs[0]
    p1.style = doc.styles["figure caption"]
    add_runs(p1, caption)
    anchor_par._p.addprevious(t._tbl)
    return t


def _scrub_app_xml(path):
    """app.xml шаблона сохранён русским Word — заменить служебную кириллицу."""
    import os
    import zipfile
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item == "docProps/app.xml":
                data = data.decode("utf8").replace("Название", "Title").encode("utf8")
            zout.writestr(item, data)
    os.replace(tmp, path)

# ------------------------------------------------------------------- builder
def build(cfg):
    shutil.copy(TEMPLATE, cfg["out"])
    doc = Document(cfg["out"])
    body = doc.element.body
    paras = doc.paragraphs

    title, abstract, kw, blocks, refs = parse_md(cfg["md"])
    if cfg.get("refs_from"):
        refs = parse_md(cfg["refs_from"])[4]

    # 1. заголовок
    clear_paragraph(paras[0]); paras[0].add_run(title)
    # 2. авторы: p2 = наш блок; p3..p5 удалить; секция авторов -> 1 колонка
    clear_paragraph(paras[2])
    for i, line in enumerate(cfg["author_block"]):
        if i:
            paras[2].add_run().add_break()
        r = paras[2].add_run(line)
        if i == 0:
            r.bold = True
    for p in paras[3:6]:
        p._p.getparent().remove(p._p)
    sect_authors = doc.sections[1]._sectPr
    cols = sect_authors.find(qn("w:cols"))
    if cols is not None and cols.get(qn("w:num")):
        del cols.attrib[qn("w:num")]

    # 3. вычистить тело между sectPr авторов и последним sectPr-абзацем
    paras = doc.paragraphs
    sect_par_idxs = [i for i, p in enumerate(paras)
                     if p._p.find(qn("w:pPr")) is not None
                     and p._p.find(qn("w:pPr")).find(qn("w:sectPr")) is not None]
    start = sect_par_idxs[-2]
    end_par = paras[sect_par_idxs[-1]]
    to_remove, started = [], False
    for el in list(body):
        if el is end_par._p:
            break
        if started and el.tag in (qn("w:p"), qn("w:tbl")):
            to_remove.append(el)
        if el is paras[start]._p:
            started = True
    for el in to_remove:
        body.remove(el)

    # 4. контент — добавляем в конец, потом переносим перед end_par
    added = []
    def track(x):
        added.append(x); return x

    pa = track(doc.add_paragraph(style="Abstract"))
    add_runs(pa, cfg["abstract_prefix"] + abstract)
    pk = track(doc.add_paragraph(style="Keywords"))
    add_runs(pk, cfg["keywords_prefix"] + kw)

    fig1_anchor_par = [None]
    for kind, payload in blocks:
        if kind == "h1":
            p = track(doc.add_paragraph(style="Heading 1")); add_runs(p, payload)
        elif kind == "h2":
            p = track(doc.add_paragraph(style="Heading 2")); add_runs(p, payload)
        elif kind == "formula":
            track(add_equation_par(doc, payload))
        else:
            p = track(doc.add_paragraph(style="Body Text"))
            add_runs(p, payload)
            if cfg["fig1_anchor"] in payload:
                fig1_anchor_par[0] = p
            for anchor, ttl, rows in cfg["tables"]:
                if anchor in payload:
                    th = track(doc.add_paragraph(style="table head"))
                    add_runs(th, ttl)
                    tbl = track(make_table(doc, rows))
                    tbl._tbl.getparent()  # spacing задаёт следующий абзац
            for anchor, img, w, cap in cfg["figures"]:
                if anchor in payload:
                    fp, cp2 = add_figure(doc, img, w, cap)
                    track(fp); track(cp2)

    rh = track(doc.add_paragraph(style="Heading 5"))
    add_runs(rh, cfg["refs_heading"])
    for _, ref in refs:
        p = track(doc.add_paragraph(style="references"))
        add_runs(p, ref)

    for item in added:
        el = item._p if hasattr(item, "_p") else item._tbl
        end_par._p.addprevious(el)

    # 5. плавающий Рис. 1
    if fig1_anchor_par[0] is None:
        raise SystemExit("fig1 anchor paragraph not found")
    add_float_fig1(doc, fig1_anchor_par[0], cfg["fig1_img"], cfg["fig1_caption"])

    # 5b. убрать хвостовую пустую секцию шаблона: двухколоночный sectPr
    # переносится на уровень тела, конец документа = конец 2-колоночной секции
    final_sectPr = body.find(qn("w:sectPr"))
    two_col_sectPr = end_par._p.find(qn("w:pPr")).find(qn("w:sectPr"))
    end_par._p.find(qn("w:pPr")).remove(two_col_sectPr)
    body.replace(final_sectPr, two_col_sectPr)
    # удалить всё после end_par (пустой хвостовой абзац) и сам end_par
    nxt = end_par._p.getnext()
    while nxt is not None and nxt.tag == qn("w:p"):
        after = nxt.getnext()
        body.remove(nxt)
        nxt = after
    body.remove(end_par._p)

    # 6. свойства документа
    cp = doc.core_properties
    cp.title = title
    cp.author = cfg["core_author"]
    cp.last_modified_by = cfg["core_author"]
    cp.subject = cfg["core_subject"]
    cp.keywords = kw.replace(";", ",")
    cp.comments = ""
    cp.category = ""

    doc.save(cfg["out"])
    _scrub_app_xml(cfg["out"])
    print("built", cfg["out"])

# -------------------------------------------------------------------- configs
def en_cfg():
    t1, r1 = parse_md_table("paper/tables/table1_corpus.md")
    t2, r2 = parse_md_table("paper/tables/table2_main_v2.md")
    t3, r3 = parse_md_table("paper/tables/table3_practice_v3.md")
    return dict(
        md="paper/piere2026_paper_en.md",
        out="paper/piere2026_paper_en.docx",
        author_block=["Elena V. Krupoderova", "Master's student",
                      "Bauman Moscow State Technical University",
                      "2nd Baumanskaya st. 5/1, 105005 Moscow, Russia",
                      "lena05k@yandex.ru"],
        abstract_prefix="Abstract—",
        keywords_prefix="Keywords—",
        refs_heading="References",
        core_author="Elena V. Krupoderova",
        core_subject="Multilingual fraud dialogue detection; validation protocol",
        tables=[
            ("Table I summarizes the corpus", t1, r1),
            ("Table II reports the full metrics", t2, r2),
            ("Table III contrasts the current", t3, r3),
        ],
        figures=[
            ("Statistical testing (Fig. 2)",
             "paper/figures/fig4_significance.png", 3.38,
             "Pairwise significance of F1 differences on the human-verified test set: observed difference (diamond), 95% paired-bootstrap confidence intervals over test dialogues (thick line) and over dialogues plus training seeds (thin line), with paired-permutation p-values; blue marks significant pairs, grey non-significant ones."),
            ("Fig. 3 plots, for all three seeds",
             "paper/figures/fig2_divergence.png", 3.38,
             "The silver ceiling in checkpoint selection: F1 of XLM-R per fine-tuning epoch on the machine-labelled silver validation set (orange) and on the human-verified gold-dev set (green); three seeds shown by line styles. Circles mark checkpoints selected on human labels (epoch 1 for all seeds); squares mark checkpoints selected on machine labels. The vertical axis is truncated to the range 0.84-1.03 for readability."),
            ("Fig. 4 shows the selection gap",
             "paper/figures/fig3_selection_gap.png", 3.38,
             "Test F1 by selection label source for five model families: green circles denote selection on human labels, orange squares selection on machine labels. The dashed line marks the silver ceiling (0.8757). mDeBERTa shows no gap; frozen-representation models show gaps below 0.002."),
            ("Fig. 5 reports the transfer matrix",
             "paper/figures/fig5_transfer.png", 3.38,
             "Cross-lingual transfer of fine-tuned XLM-R: median F1 over three seeds per training configuration (rows, with training size) and test language (columns). Asterisks mark zero-shot cells, where the target language is absent from both training and selection; the framed cell is the single large anomaly (English to Chinese). Medians are used because 6 of 87 runs collapsed to a degenerate predictor."),
        ],
        fig1_anchor="We consider binary classification",
        fig1_img="paper/figures/fig1_pipeline.png",
        fig1_caption="The evaluation pipeline: one corpus and three model families, two checkpoint-selection paths (human-verified gold-dev versus machine-labelled silver-val), and the two resulting answers on the same human-verified test set.",
    )

RU_T1 = ("Статистика корпуса", [
    ["Параметр", "Значение"],
    ["Диалоги (скам / безобидные)", "2000 (1001 / 999)"],
    ["Языки ru / en / zh", "667 / 667 / 666"],
    ["Трудные отрицательные", "348"],
    ["Реплик в диалоге (среднее)", "6-14 (10,04)"],
    ["Сценарии", "19"],
    ["Разбиение train / val / test", "1392 / 205 / 403"],
    ["Верифицировано (dev / test)", "100 / 303"],
    ["Межаннотаторская каппа (бинарная, n = 98)", "0,804"],
    ["Согласие silver-gold (тест)", "0,8614 (каппа 0,717)"],
])
RU_T2 = ("Основное сравнение на верифицированном тесте (gold v2, n = 303); среднее по 3 сидам", [
    ["Модель", "F1", "Точн.", "Полн.", "Каппа", "Согл. silver"],
    ["Константа «всегда скам»", "0,7658", "0,6205", "1,0000", "0,0000", "-"],
    ["TF-IDF + LogReg", "0,8713", "0,9675", "0,7926", "0,7084", "0,9868"],
    ["e5 + LightGBM", "0,8789 ± 0,0015", "0,9783", "0,7979", "0,7264", "0,9890"],
    ["mDeBERTa-v3-base", "0,8906 ± 0,0383", "0,8941", "0,8954", "0,7038", "0,8493"],
    ["Ключевые слова (20/язык)", "0,9196", "0,8714", "0,9734", "0,7671", "0,7954"],
    ["XLM-R base", "0,9344 ± 0,0097", "0,9349", "0,9344", "0,8272", "0,8372"],
])
RU_T3 = ("Текущая и предлагаемая практика валидации (числа данного исследования, XLM-R)", [
    ["Аспект", "Текущая практика", "Предлагаемая"],
    ["Источник меток для отбора", "машинные, silver-val (205 диалогов)",
     "человеческие, gold-dev (100 диалогов)"],
    ["Видимое согласие с валидацией", "F1 = 1,0000", "F1 = 0,9106"],
    ["F1 на верифицированном тесте (n = 303)", "0,8757", "0,9344"],
    ["Цена в размеченных человеком диалогах", "0", "100"],
])

def ru_cfg():
    return dict(
        md="paper/piere2026_report_ru.md",
        out="paper/piere2026_report_ru.docx",
        refs_from="paper/piere2026_paper_en.md",
        author_block=["Круподерова Елена Вячеславовна", "магистрант",
                      "МГТУ им. Н. Э. Баумана",
                      "2-я Бауманская ул., 5/1, 105005 Москва, Россия",
                      "lena05k@yandex.ru"],
        abstract_prefix="Аннотация—",
        keywords_prefix="Ключевые слова—",
        refs_heading="Список источников",
        core_author="Круподерова Елена Вячеславовна",
        core_subject="Многоязычная детекция мошеннических диалогов; протокол валидации",
        tables=[
            ("Таблица I суммирует корпус", *RU_T1),
            ("Таблица II приводит полные метрики", *RU_T2),
            ("Таблица III противопоставляет", *RU_T3),
        ],
        figures=[
            ("Статистическая проверка (рис. 2)",
             "paper/figures/fig4_significance.png", 3.38,
             "Парная значимость различий F1 на верифицированном тесте: наблюдаемая разница (ромб), 95-процентные доверительные интервалы парного бутстрапа по тестовым диалогам (толстая линия) и по диалогам вместе с сидами обучения (тонкая), с перестановочными p-значениями; синим показаны значимые пары, серым — незначимые."),
            ("Рис. 3 показывает для всех трёх сидов",
             "paper/figures/fig2_divergence.png", 3.38,
             "Серебряный потолок при отборе чекпоинта: F1 модели XLM-R по эпохам дообучения на машинно-размеченной серебряной валидации (оранжевый) и на верифицированном человеком gold-dev (зелёный); три сида показаны стилями линий. Кружки — чекпоинты, отобранные по человеческим меткам (эпоха 1 на всех сидах); квадраты — отобранные по машинным. Ось ординат усечена до 0,84-1,03 для читаемости."),
            ("Рис. 4 показывает разрыв отбора",
             "paper/figures/fig3_selection_gap.png", 3.38,
             "Тестовый F1 в зависимости от источника меток при отборе для пяти семейств моделей: зелёные кружки — отбор по человеческим меткам, оранжевые квадраты — по машинным. Пунктир — серебряный потолок (0,8757). У mDeBERTa разрыва нет; у моделей над замороженными представлениями разрыв меньше 0,002."),
            ("Рис. 5 приводит матрицу переноса",
             "paper/figures/fig5_transfer.png", 3.38,
             "Межъязыковой перенос дообученной XLM-R: медианный F1 по трём сидам для каждой конфигурации обучения (строки, с объёмом обучения) и тестового языка (столбцы). Звёздочки — клетки нулевого переноса (целевой язык отсутствует и в обучении, и в отборе); рамкой обведена единственная крупная аномалия (английский → китайский). Медианы использованы потому, что 6 из 87 прогонов схлопнулись в вырожденный предсказатель."),
        ],
        fig1_anchor="Рассматривается бинарная классификация",
        fig1_img="paper/figures/fig1_pipeline.png",
        fig1_caption="Конвейер оценки: один корпус и три семейства моделей, два пути отбора чекпоинта (по человеческим меткам gold-dev и по машинным silver-val) и два получающихся ответа на одном и том же верифицированном тесте.",
    )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("en", "both"):
        build(en_cfg())
    if which in ("ru", "both"):
        build(ru_cfg())
