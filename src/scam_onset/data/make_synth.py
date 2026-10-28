#!/usr/bin/env python3
"""
Генератор СИНТЕТИЧЕСКОГО пилотного набора для теста и разработки.

Это hand-authored фикстуры (не LLM-генерация стадии 1 конвейера) — выдуманные
размеченные диалоги в формате проекта, чтобы разрабатывать metrics/losses/models
без аннотаторов и без реальных данных. Чувствительное — плейсхолдеры.

Пишет:
  data/synth/pilot_dialogues.jsonl  — текст диалогов + gold-метки
  data/synth/pilot_gold.csv         — те же метки в формате харнесса
"""
import json, csv, pathlib

# (item_id, id, lang, type, turns[(role,text)], is_scam, (phish,money,other), other_note, onset, hn)
A, T = "atk", "tgt"
DIALOGS = [
 (1,"syn_ru_phish_sms","ru","phishing",[
   (A,"Ваша посылка задержана на таможне. Оплатите сбор (AMOUNT) здесь: (URL), чтобы её выпустили.")],
   1,(1,1,0),"",1,0),
 (2,"syn_ru_bank_imperson","ru","impersonation",[
   (A,"Здравствуйте, это служба безопасности вашего банка."),
   (T,"Здравствуйте, а что случилось?"),
   (A,"Зафиксирована попытка списания. Подтвердите данные, иначе счёт заблокируют."),
   (A,"Продиктуйте код из СМС и номер карты.")],
   1,(0,1,1),"impersonation",1,0),
 (3,"syn_ru_romance","ru","romance",[
   (A,"Не могу перестать думать о тебе со вчера."),
   (T,"Мне тоже приятно, хотя мы едва знакомы."),
   (A,"Ты особенная, я редко кому открываюсь."),
   (A,"Я сейчас на контракте на платформе в море, связь плохая."),
   (A,"За границей заблокировали карту, скинешь (AMOUNT) на пару дней? Верну.")],
   1,(0,1,1),"romance",4,0),
 (4,"syn_ru_wrongnum_invest","ru","investment",[
   (A,"Привет, (NAME)! Столик на восемь подтверждён?"),
   (T,"Вы ошиблись номером."),
   (A,"Ой, простите! Хорошего дня."),
   (A,"Раз списались — приятно пообщаться. Я аналитик, недавно переехала."),
   (A,"Покажу площадку, где стабильно делаю 15% в месяц, если интересно.")],
   1,(0,1,1),"investment",4,0),
 (5,"syn_ru_refund_support","ru","refund",[
   (A,"Здравствуйте! Поддержка «МаркетПлюс» по вашему возврату за чайник."),
   (T,"Да, я оформлял возврат вчера."),
   (A,"Возврат завис. Установите приложение «МП-Возврат» и продиктуйте код из СМС — подтвердим карту."),
   (A,"Поторопитесь, заявка аннулируется через 15 минут.")],
   1,(0,1,1),"remote_access",1,0),
 (6,"syn_ru_fakejob","ru","fake_job",[
   (A,"Поздравляем, вы приняты на удалёнку по вводу данных, (AMOUNT)/нед."),
   (T,"Отлично, что дальше?"),
   (A,"Оплатите взнос за активацию аккаунта (AMOUNT).")],
   1,(0,1,1),"fake_job",1,0),
 (7,"syn_ru_prize","ru","prize",[
   (A,"Вы выиграли сертификат на (AMOUNT)! Заберите в течение часа: (URL)")],
   1,(1,0,0),"",1,0),
 (8,"syn_ru_techsupport","ru","tech_support",[
   (A,"Это поддержка (BRAND). Ваш ПК заражён и рассылает спам."),
   (A,"Установите (APP) и продиктуйте код, я всё почищу.")],
   1,(0,0,1),"tech_support",1,0),
 (9,"syn_ru_boss","ru","impersonation",[
   (A,"Ты на месте? Нужен срочный перевод, тихо, до совещания."),
   (T,"Да, какая сумма?"),
   (A,"(AMOUNT) на счёт подрядчика, подтвержу потом. Между нами.")],
   1,(0,1,1),"impersonation",1,0),
 (10,"syn_ru_bank_true","ru","benign_bank",[
   (A,"БАНК: оплата (AMOUNT) в «Продукты» проведена. Если это не вы — позвоните по номеру на карте.")],
   0,(0,0,0),"",None,1),
 (11,"syn_ru_delivery_true","ru","benign_delivery",[
   (A,"Ваш заказ (NUM) отправлен, доставка завтра. Отслеживание: (URL)")],
   0,(0,0,0),"",None,1),
 (12,"syn_ru_fraud_warning_true","ru","benign_warning",[
   (A,"Банк: заблокирована подозрительная операция (AMOUNT). Если это не вы — никому не сообщайте коды и позвоните по номеру на обороте карты.")],
   0,(0,0,0),"",None,1),
 (13,"syn_ru_friend_loan","ru","benign_loan",[
   (A,"Слушай, займёшь (AMOUNT) до пятницы? Верну."),
   (T,"Да, сейчас скину.")],
   0,(0,0,0),"",None,0),
 (14,"syn_ru_marketing_true","ru","benign_marketing",[
   (A,"ПОСЛЕДНИЙ ШАНС — скидка 50% только сегодня! (URL)"),
   (T,"Спасибо, нет.")],
   0,(0,0,0),"",None,1),
 (15,"syn_en_phish","en","phishing",[
   (A,"Your package is held at customs. Pay the (AMOUNT) fee here to release it: (URL)")],
   1,(1,1,0),"",1,0),
 (16,"syn_en_bank","en","impersonation",[
   (A,"Hello, this is your bank's security department."),
   (T,"Hi, what's going on?"),
   (A,"We detected a suspicious charge. Confirm your details now or your account will be locked."),
   (A,"Read me the SMS code and your card number.")],
   1,(0,1,1),"impersonation",1,0),
 (17,"syn_en_romance","en","romance",[
   (A,"I haven't been able to stop thinking about you since yesterday."),
   (T,"That's sweet, though we barely know each other."),
   (A,"You're special. I rarely open up to anyone."),
   (A,"I'm on a contract on an offshore platform right now, signal is bad."),
   (A,"My card got blocked abroad, could you send (AMOUNT) for a few days? I'll pay back.")],
   1,(0,1,1),"romance",4,0),
 (18,"syn_en_job","en","fake_job",[
   (A,"Congratulations, you're hired for remote data entry, (AMOUNT)/week."),
   (T,"Great, what's next?"),
   (A,"Please pay the (AMOUNT) account activation fee.")],
   1,(0,1,1),"fake_job",1,0),
 (19,"syn_en_delivery_true","en","benign_delivery",[
   (A,"Your order (NUM) has shipped, arriving tomorrow. Track: (URL)")],
   0,(0,0,0),"",None,1),
 (20,"syn_en_bank_true","en","benign_bank",[
   (A,"BANK: payment of (AMOUNT) at Grocery completed. If this wasn't you, call the number on your card.")],
   0,(0,0,0),"",None,1),
 (21,"syn_zh_phish","zh","phishing",[
   (A,"您的包裹被海关扣留。请点击此链接支付(AMOUNT)手续费以放行：(URL)")],
   1,(1,1,0),"",1,0),
 (22,"syn_zh_bank","zh","impersonation",[
   (A,"您好，这里是您银行的安全部门。"),
   (T,"您好，出什么事了？"),
   (A,"我们检测到一笔可疑交易。请立即确认您的信息，否则账户将被冻结。"),
   (A,"请把短信验证码和银行卡号告诉我。")],
   1,(0,1,1),"impersonation",1,0),
 (23,"syn_zh_romance","zh","romance",[
   (A,"从昨天起我就一直在想你。"),
   (T,"我也很高兴，虽然我们才刚认识。"),
   (A,"你很特别，我很少对人敞开心扉。"),
   (A,"我现在在海上的平台上工作，信号不好。"),
   (A,"我在国外银行卡被冻结了，能先借我(AMOUNT)几天吗？我会还。")],
   1,(0,1,1),"romance",4,0),
 (24,"syn_zh_delivery_true","zh","benign_delivery",[
   (A,"您的订单(NUM)已发货，明天送达。物流查询：(URL)")],
   0,(0,0,0),"",None,1),
]

def build(root="."):
    root = pathlib.Path(root); out = root / "data" / "synth"; out.mkdir(parents=True, exist_ok=True)
    subs = ["phishing_link","money_request","other"]
    jl = out / "pilot_dialogues.jsonl"; gc = out / "pilot_gold.csv"
    with open(jl,"w",encoding="utf-8") as fj, open(gc,"w",newline="",encoding="utf-8") as fc:
        w = csv.writer(fc)
        w.writerow(["item_id","T","round","is_scam","sub_phish","sub_money","sub_other","onset","hard_negative"])
        for iid,slug,lang,typ,turns,scam,(p,m,o),note,onset,hn in DIALOGS:
            rec = {"item_id":iid,"id":slug,"lang":lang,"scam_type":typ,
                   "turns":[{"i":i+1,"role":r,"text":t} for i,(r,t) in enumerate(turns)],
                   "is_scam":scam,
                   "subtypes":[s for s,f in zip(subs,(p,m,o)) if f],
                   "other_note":note,"onset":onset,"hard_negative":hn}
            fj.write(json.dumps(rec,ensure_ascii=False)+"\n")
            w.writerow([iid,len(turns),"synth",scam,p,m,o,("" if onset is None else onset),hn])
    return jl, gc

if __name__ == "__main__":
    jl, gc = build(".")
    n = len(DIALOGS); scam = sum(d[5] for d in DIALOGS); hn = sum(d[9] for d in DIALOGS)
    langs = sorted({d[2] for d in DIALOGS})
    print(f"написано: {jl}  и  {gc}")
    print(f"диалогов: {n} | scam: {scam} | benign: {n-scam} | hard_neg: {hn} | языки: {langs}")
