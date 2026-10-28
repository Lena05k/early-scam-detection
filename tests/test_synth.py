import json, csv, subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def _ensure_built():
    jl = ROOT/"data"/"synth"/"pilot_dialogues.jsonl"
    if not jl.exists():
        subprocess.run([sys.executable, "src/scam_onset/data/make_synth.py"], cwd=ROOT, check=True)
    return [json.loads(l) for l in open(jl, encoding="utf-8")]

def test_benign_has_empty_onset_and_no_subtypes():
    for d in _ensure_built():
        if d["is_scam"] == 0:
            assert d["onset"] is None, d["id"]           # benign -> ∅
            assert d["subtypes"] == [], d["id"]

def test_scam_onset_in_range_and_on_attacker_turn():
    for d in _ensure_built():
        if d["is_scam"] == 1:
            T = len(d["turns"])
            assert d["onset"] is not None and 1 <= d["onset"] <= T, d["id"]
            role_at_onset = d["turns"][d["onset"]-1]["role"]
            assert role_at_onset == "atk", d["id"]        # онсет привязан к атакующей стороне
            assert d["subtypes"], d["id"]                 # scam -> есть хотя бы один подтип

def test_multilingual_and_variety():
    data = _ensure_built()
    assert {d["lang"] for d in data} == {"ru","en","zh"}
    assert any(d["hard_negative"] == 1 for d in data)     # есть hard negatives
    assert any(d["is_scam"] == 1 and d["onset"] > 1 for d in data)  # есть отложенный онсет

def test_csv_matches_jsonl():
    data = _ensure_built()
    rows = list(csv.DictReader(open(ROOT/"data"/"synth"/"pilot_gold.csv", encoding="utf-8")))
    assert len(rows) == len(data)
    by_id = {d["item_id"]: d for d in data}
    for r in rows:
        d = by_id[int(r["item_id"])]
        assert int(r["is_scam"]) == d["is_scam"]
        assert (r["onset"] == "" ) == (d["onset"] is None)
