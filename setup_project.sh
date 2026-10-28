#!/usr/bin/env bash
set -euo pipefail

echo ">> создаю дерево каталогов"
mkdir -p src/scam_onset/iaa src/scam_onset/data src/scam_onset/models \
         src/scam_onset/metrics src/scam_onset/losses \
         configs scripts tests docs \
         data/raw data/skeletons data/synth data/annotations data/augmented data/splits

echo ">> __init__.py + .gitkeep"
for d in src/scam_onset src/scam_onset/iaa src/scam_onset/data \
         src/scam_onset/models src/scam_onset/metrics src/scam_onset/losses; do
  touch "$d/__init__.py"
done
for d in data/raw data/skeletons data/synth data/augmented data/splits data/annotations; do
  touch "$d/.gitkeep"
done

echo ">> раскладываю файлы по местам (если найдены рядом)"
move() { [ -f "$1" ] && { mkdir -p "$(dirname "$2")"; mv -f "$1" "$2"; echo "   $1 -> $2"; } || true; }

move scam_onset_methodology_v1.md docs/methodology_v1.md
move calibration_protocol_v1.md   docs/calibration_protocol_v1.md
move annotation_guidelines_v1.md  docs/annotation_guidelines_v1.md
move iaa_harness.py               src/scam_onset/iaa/harness.py
move calibration_items_gold.csv   data/annotations/calibration_items_gold.csv
move annotator_demo.csv           data/annotations/annotator_demo.csv
move calibration_report_demo.txt  data/annotations/calibration_report_demo.txt
move run_iaa.py                   scripts/run_iaa.py
move test_iaa.py                  tests/test_iaa.py
move default.yaml                 configs/default.yaml

echo ""
echo ">> готово."
