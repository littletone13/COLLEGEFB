import os, time
from datetime import datetime
import pandas as pd
import requests

TWILIO_LOOKUP_URL = "https://lookups.twilio.com/v2/PhoneNumbers/{phone}?type=carrier"

def normalize(num: str) -> str:
    digits = "".join(ch for ch in str(num) if ch.isdigit())
    if not digits:
        return ""
    if len(digits) == 10:
        return f"+1{digits}"
    if digits.startswith("1") and len(digits) == 11:
        return f"+{digits}"
    if str(num).startswith("+"):
        return str(num)
    return "+" + digits

def twilio_lookup(phone: str, sid: str, token: str):
    url = TWILIO_LOOKUP_URL.format(phone=phone)
    try:
        resp = requests.get(url, auth=(sid, token), timeout=20)
        if resp.status_code != 200:
            return {}
        data = resp.json() or {}
        carrier = data.get("carrier", {}) or {}
        return {
            "carrier_name": carrier.get("name", ""),
            "line_type": carrier.get("type", ""),
            "sms_capable": True if carrier.get("type","") == "mobile" else False,
            "verification_source": "twilio_lookup"
        }
    except Exception as e:
        print(f"[!] Twilio lookup error for {phone}: {e}")
        return {}

def numverify_lookup(phone: str, key: str):
    try:
        url = "http://apilayer.net/api/validate"
        resp = requests.get(url, params={"access_key": key, "number": phone}, timeout=20)
        if resp.status_code != 200:
            return {}
        data = resp.json() or {}
        lt = data.get("line_type") or ""
        return {
            "carrier_name": data.get("carrier",""),
            "line_type": lt,
            "sms_capable": True if lt == "mobile" else False,
            "verification_source": "numverify"
        }
    except Exception as e:
        print(f"[!] Numverify error for {phone}: {e}")
        return {}

def verify_master_numbers(excel_path: str, sleep: float = 0.25):
    df_master = pd.read_excel(excel_path, sheet_name="Master")
    for c in ["carrier_name","line_type","sms_capable","verification_date","verification_source"]:
        if c not in df_master.columns:
            df_master[c] = ""
    tw_sid = os.getenv("TWILIO_SID","")
    tw_token = os.getenv("TWILIO_AUTH_TOKEN","")
    nv_key = os.getenv("NUMVERIFY_KEY","")
    PHONE_FIELDS = ["owner_mobile","phone_primary","owner_phone","phone_alt"]

    sms_queue_rows = []
    for idx, row in df_master.iterrows():
        raw_candidates = [row.get(f) for f in PHONE_FIELDS if pd.notna(row.get(f)) and str(row.get(f)).strip()]
        best = raw_candidates[0] if raw_candidates else ""
        phone_norm = normalize(best)
        if not phone_norm:
            continue
        info = {}
        if tw_sid and tw_token:
            info = twilio_lookup(phone_norm, tw_sid, tw_token)
        if not info and nv_key:
            info = numverify_lookup(phone_norm, nv_key)
        if info:
            df_master.at[idx, "carrier_name"] = info.get("carrier_name","")
            df_master.at[idx, "line_type"] = info.get("line_type","")
            df_master.at[idx, "sms_capable"] = info.get("sms_capable", False)
            df_master.at[idx, "verification_source"] = info.get("verification_source","")
            df_master.at[idx, "verification_date"] = datetime.utcnow().strftime("%Y-%m-%d")
        if info.get("sms_capable"):
            sms_queue_rows.append({
                "company_id": row.get("company_id",""),
                "company_name": row.get("company_name",""),
                "contact_name": row.get("owner_name",""),
                "phone": phone_norm,
                "line_type": info.get("line_type",""),
                "carrier_name": info.get("carrier_name",""),
                "sms_capable": True,
                "template_id": "intro_partner_v1",
                "message_preview": f"Hey {row.get('owner_name','there')}, we offer fast stump grinding for overflow jobs in STL. Interested in a quick quote/partner rate? —{row.get('company_name','')}",
                "approved": "",
                "sent": "",
                "send_timestamp": "",
                "twilio_sid": "",
                "notes": ""
            })
        time.sleep(sleep)

    # Update SMS_Queue sheet
    try:
        df_q = pd.read_excel(excel_path, sheet_name="SMS_Queue")
    except Exception:
        df_q = pd.DataFrame()
    import pandas as pd
    df_new = pd.DataFrame(sms_queue_rows)
    if not df_q.empty and not df_new.empty:
        combined = pd.concat([df_q, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["phone"], keep="first")
    else:
        combined = df_new if not df_new.empty else df_q

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_master.to_excel(writer, index=False, sheet_name="Master")
        combined.to_excel(writer, index=False, sheet_name="SMS_Queue")

    print(f"[✓] Verified {len(df_master)} rows. SMS_Queue now has {len(combined)} rows (none sent).")
