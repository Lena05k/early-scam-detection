# -*- coding: utf-8 -*-
"""Слепой калибровочный пакет из JSONL-диалогов: тексты без меток + пустые бланки."""
import json, csv, pathlib, argparse
def build(jsonl, out_md, ann_dir, who="ABCDE"):
    ds=[json.loads(l) for l in open(jsonl,encoding="utf-8")]
    role={"atk":"A","tgt":"B"}
    with open(out_md,"w",encoding="utf-8") as f:
        f.write("# Синтетические диалоги для разметки\n\n")
        f.write("Размечай сверху вниз, реплику за репликой. Останавливайся на ПЕРВОЙ реплике "
                "атакующего (A) с наблюдаемым обманом (a–d из памятки). Не читай весь диалог заранее.\n\n---\n\n")
        for d in ds:
            f.write(f"## Диалог {d['item_id']}\n\n")
            for t in d["turns"]:
                f.write(f"[{t['i']}] {role.get(t['role'],t['role'])}: {t['text']}\n\n")
            f.write(f"Метки {d['item_id']}: is_scam = ___ | subtypes = ___ | onset = ___\n\n---\n\n")
    ad=pathlib.Path(ann_dir); ad.mkdir(parents=True,exist_ok=True)
    for w in who:
        with open(ad/f"synth_annotator_{w}.csv","w",newline="",encoding="utf-8") as f:
            wr=csv.writer(f); wr.writerow(["item_id","annotator","is_scam","sub_phish","sub_money","sub_other","onset"])
            for d in ds: wr.writerow([d["item_id"],w,"","","","",""])
    print(f"слепой пакет: {out_md} + бланки synth_annotator_[{who}].csv в {ann_dir}")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--jsonl",default="data/synth/pilot_dialogues.jsonl")
    p.add_argument("--md",default="docs/dialogs_to_annotate_synth.md")
    p.add_argument("--ann",default="data/annotations")
    a=p.parse_args(); build(a.jsonl,a.md,a.ann)
