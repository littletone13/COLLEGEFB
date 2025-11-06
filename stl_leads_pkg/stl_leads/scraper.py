import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"

def _http_json(url: str, method: str = "GET", headers: Optional[Dict[str,str]] = None, json_body: Optional[Dict]=None, params: Optional[Dict]=None, retries: int = 3, backoff: float = 1.5) -> Dict:
    for attempt in range(retries):
        try:
            if method.upper() == "POST":
                r = requests.post(url, headers=headers, json=json_body, params=params, timeout=30)
            else:
                r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = backoff ** (attempt+1)
                time.sleep(wait)
                continue
            print(f"[!] HTTP {r.status_code}: {r.text[:300]}")
            break
        except requests.RequestException as e:
            print(f"[!] Request error: {e}")
            time.sleep(backoff ** (attempt+1))
    return {}

def places_text_search(api_key: str, query: str, lat: float, lng: float, radius: int, region: str = "us", max_results: int = 300) -> List[Dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount,places.businessStatus,nextPageToken",
    }
    body = {
        "textQuery": query,
        "regionCode": region,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius
            }
        }
    }
    results = []
    while True:
        data = _http_json(PLACES_TEXT_SEARCH_URL, method="POST", headers=headers, json_body=body)
        if not data:
            break
        places = data.get("places", [])
        results.extend(places)
        token = data.get("nextPageToken")
        if not token or len(results) >= max_results:
            break
        body["pageToken"] = token
        time.sleep(2.0)
    return results[:max_results]

def place_details(api_key: str, place_id: str, fields: str):
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": fields
    }
    url = PLACES_DETAILS_URL.format(place_id=place_id)
    return _http_json(url, headers=headers)

def flatten_place(place: Dict[str,Any]) -> Dict[str,Any]:
    dn = place.get("displayName")
    name = dn.get("text") if isinstance(dn, dict) else dn
    loc = place.get("location") or {}
    return {
        "place_id": place.get("id"),
        "company_name": name,
        "display_name": name,
        "business_status": place.get("businessStatus"),
        "formatted_address": place.get("formattedAddress"),
        "latitude": (loc.get("latitude") if isinstance(loc, dict) else None),
        "longitude": (loc.get("longitude") if isinstance(loc, dict) else None),
        "rating": place.get("rating"),
        "user_rating_count": place.get("userRatingCount"),
        "types": ",".join(place.get("types", []) or []),
        "fetch_timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

DEFAULT_FIELDS = ",".join([
    "id",
    "displayName",
    "formattedAddress",
    "internationalPhoneNumber",
    "nationalPhoneNumber",
    "websiteUri",
    "types",
    "rating",
    "userRatingCount",
    "businessStatus",
    "location",
])

def enrich_with_details(api_key: str, place_id: str) -> Dict[str,Any]:
    details = place_details(api_key, place_id, DEFAULT_FIELDS)
    if not details:
        return {}
    return {
        "international_phone_number": details.get("internationalPhoneNumber",""),
        "national_phone_number": details.get("nationalPhoneNumber",""),
        "website_uri": details.get("websiteUri",""),
    }

def append_to_excel(excel_path: str, sheet: str, df_new: pd.DataFrame, key_cols: list):
    try:
        existing = pd.read_excel(excel_path, sheet_name=sheet, dtype=str)
    except Exception:
        existing = pd.DataFrame(columns=df_new.columns)
    for col in df_new.columns:
        if col not in existing.columns:
            existing[col] = None
    for col in existing.columns:
        if col not in df_new.columns:
            df_new[col] = None
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="first")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        combined.to_excel(writer, index=False, sheet_name=sheet)
    return combined

def scrape_places_to_excel(excel_path: str, api_key: str, keywords: list, lat: float, lng: float, radius: int, region: str, sleep: float, max_results_per_keyword: int):
    all_rows = []
    for q in keywords:
        print(f"[i] Searching: {q}")
        places = places_text_search(api_key, q, lat, lng, radius, region=region, max_results=max_results_per_keyword)
        print(f"[i] Found {len(places)} candidates for '{q}'")
        for p in places:
            base = flatten_place(p)
            details = enrich_with_details(api_key, base["place_id"])
            time.sleep(sleep)
            row = {**base, **details, "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{base['place_id']}"}
            all_rows.append(row)
    if all_rows:
        df_places = pd.DataFrame(all_rows)
        append_to_excel(excel_path, "Raw_Places", df_places, key_cols=["place_id"])
        print(f"[✓] Appended {len(df_places)} rows to Raw_Places")
    else:
        print("[!] No places gathered.")
