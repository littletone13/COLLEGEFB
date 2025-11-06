#!/usr/bin/env python3
import argparse

from .scraper import scrape_places_to_excel
from .verifier import verify_master_numbers

def main():
    parser = argparse.ArgumentParser(prog="stl-leads", description="Scrape (Google Places) + Verify (Twilio/Numverify) + Queue SMS (no send).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="Scrape Google Places and append to Raw_Places")
    s.add_argument("--excel", required=True)
    s.add_argument("--google_api_key", required=True)
    s.add_argument("--keywords", default="tree service,arborist,tree trimming,landscaper")
    s.add_argument("--lat", type=float, default=38.6270)
    s.add_argument("--lng", type=float, default=-90.1994)
    s.add_argument("--radius", type=int, default=50000)
    s.add_argument("--region", default="us")
    s.add_argument("--sleep", type=float, default=0.2)
    s.add_argument("--max_results_per_keyword", type=int, default=200)

    v = sub.add_parser("verify", help="Verify phone numbers in Master and build SMS_Queue (no texting)")
    v.add_argument("--excel", required=True)
    v.add_argument("--sleep", type=float, default=0.25)

    args = parser.parse_args()

    if args.cmd == "scrape":
        keywords = [q.strip() for q in args.keywords.split(",") if q.strip()]
        scrape_places_to_excel(
            excel_path=args.excel,
            api_key=args.google_api_key,
            keywords=keywords,
            lat=args.lat,
            lng=args.lng,
            radius=args.radius,
            region=args.region,
            sleep=args.sleep,
            max_results_per_keyword=args.max_results_per_keyword
        )
    elif args.cmd == "verify":
        verify_master_numbers(args.excel, sleep=args.sleep)

if __name__ == "__main__":
    main()
