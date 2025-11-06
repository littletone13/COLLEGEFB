
from __future__ import annotations
import argparse, json, math, pandas as pd
from pathlib import Path
from pricing import edge_vs_assumed, ev_over, breakeven_prob, fair_price_decimal

ASSUMED_DEFAULT = "receptions=1.83,rec_yds=1.87,rush_yds=1.87,rush_att=1.87,pass_yds=1.87,pass_comp=1.87"

def parse_assumed_map(s: str) -> dict[str,float]:
    out: dict[str,float] = {}
    for part in s.split(","):
        if not part.strip(): continue
        k,v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out

def _num(x):
    try:
        return None if x is None or x=="" else float(x)
    except Exception:
        return None


def _clean_value(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

def enrich_row(row: dict, assumed_map: dict[str,float]) -> dict:
    line_value = row.get("line")
    if line_value is None or (isinstance(line_value, float) and math.isnan(line_value)) or pd.isna(line_value):
        return row
    p = row.get("prob_over")
    lt = row.get("line_type")
    if p is None or lt is None:
        return row
    price = row.get("price_over")
    if price is not None:
        row["ev_over"] = ev_over(p, price)
        row["breakeven_prob"] = breakeven_prob(price)
        row["fair_price_decimal"] = fair_price_decimal(p)
        row["price_source"] = "posted"
    else:
        assumed = assumed_map.get(lt)
        if assumed is not None:
            e = edge_vs_assumed(p, assumed)
            row.update(e)
            row["ev_over"] = e["ev_over"]
            row["price_over"] = e["assumed_price_decimal"]
            row["price_source"] = "assumed"
        else:
            row["fair_price_decimal"] = fair_price_decimal(p)
            from pricing import decimal_to_american
            row["fair_price_american"] = decimal_to_american(row["fair_price_decimal"])
            row["price_source"] = None
    return row

def write_player_prop_html(df: pd.DataFrame, out_path: str, template_path: str,
                           title_suffix: str, model_tag: str, assumed_string: str):
    want = ["player","team","opponent","line_type","line","prob_over","mean","p25","p50","p75","p95","book","price_over","notes","kickoff_iso","ev_over","breakeven_prob","fair_price_decimal","fair_price_american","price_source"]
    rows = []
    for _, r in df.iterrows():
        d = {k: r.get(k) for k in want if k in r}
        for k in ["line","prob_over","mean","p25","p50","p75","p95","price_over","ev_over","breakeven_prob","fair_price_decimal"]:
            if k in d:
                d[k] = _num(d[k])
        d = {k: _clean_value(v) for k, v in d.items()}
        rows.append(d)

    data_json = json.dumps(rows, separators=(",",":"))
    meta = json.dumps({"title_suffix": title_suffix, "model_tag": model_tag, "assumed_string": assumed_string})

    html = Path(template_path).read_text(encoding="utf-8")
    html = html.replace('<script id="data" type="application/json">[]</script>',
                        f'<script id="data" type="application/json">{data_json}</script>')
    html = html.replace('<script id="meta" type="application/json">{"title_suffix":"Report","model_tag":"Model","assumed_string":""}</script>',
                        f'<script id="meta" type="application/json">{meta}</script>')

    Path(out_path).write_text(html, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_html", required=True)
    ap.add_argument("--template", default="template_report.html")
    ap.add_argument("--title", default="Week 10 (Late)")
    ap.add_argument("--model-tag", default="CFB v0.7 - wk10L")
    ap.add_argument("--assumed", default=ASSUMED_DEFAULT)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    assumed_map = parse_assumed_map(args.assumed)
    recs = df.to_dict("records")
    recs = [enrich_row(r, assumed_map) for r in recs]
    df2 = pd.DataFrame(recs)

    if "ev_over" in df2.columns and df2["ev_over"].notna().any():
        df2 = df2.sort_values("ev_over", ascending=False)
    elif set(["mean","line"]).issubset(df2.columns):
        df2["margin"] = df2["mean"] - df2["line"]
        df2 = df2.sort_values("margin", ascending=False)

    write_player_prop_html(df2, args.out_html, args.template, args.title, args.model_tag, args.assumed)

if __name__ == "__main__":
    main()
