import csv, glob, re
HEADER=["item_id","annotator","is_scam","sub_phish","sub_money","sub_other","onset"]
out=[]
for f in sorted(glob.glob('data/annotations/synth_annotator_[A-Z].csv')):
    who=re.search(r'synth_annotator_([A-Z])', f).group(1)
    rows=list(csv.reader(open(f, encoding='utf-8-sig')))
    hi=next((k for k,r in enumerate(rows) if r and r[0].strip().lower()=='item_id'), None)
    if hi is None: print('ПРОПУСК:', f); continue
    for r in rows[hi+1:]:
        if not r or not r[0].strip().isdigit(): continue
        iid,_a,s,p,m,o,on=[c.strip() for c in (r+['']*7)[:7]]
        if s=='0':            # benign -> подтипы 0, онсет пусто
            p=m=o='0'; on=''
        elif on=='0':          # scam с онсетом 0 -> тоже пусто (не бывает реплики 0)
            on=''
        out.append([iid,who,s,p,m,o,on])
with open('data/annotations/synth_calib.csv','w',newline='',encoding='utf-8') as of:
    w=csv.writer(of); w.writerow(HEADER); w.writerows(out)
print('склеено', len(out), 'строк (benign-онсет нормализован) -> synth_calib.csv')
