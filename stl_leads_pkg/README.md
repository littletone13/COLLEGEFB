# stl-leads

One CLI to **scrape** St. Louis landscaper & tree-service businesses (Google Places API),
then **verify** numbers (Twilio Lookup / Numverify) and build an **SMS_Queue** (no auto-sending).

## Install (editable)

```bash
pip install -e ./stl_leads_pkg
```

## Usage

### 1) Scrape (writes to `Raw_Places` in your Excel)
```bash
stl-leads scrape \
  --excel stl_tree_landscaping_leads_template.xlsx \
  --google_api_key YOUR_KEY \
  --keywords "tree service,arborist,tree trimming,landscaper" \
  --lat 38.6270 --lng -90.1994 --radius 50000
```

### 2) Verify (classifies phone numbers, fills Master columns, builds `SMS_Queue`)
```bash
export TWILIO_SID=ACxxxxxxxxxxxxxxxx
export TWILIO_AUTH_TOKEN=xxxxxxxxxxxx
# Optional fallback:
# export NUMVERIFY_KEY=xxxxxxxxxxxx

stl-leads verify --excel stl_tree_landscaping_leads_template.xlsx
```

> Nothing is texted. You will later mark `approved = Y` in SMS_Queue and use a sender script if desired.

## Notes
- Uses only official APIs (keeps you ToS-compliant).
- Re-runnable: de-dupes by `place_id` in Raw_Places.
- Make sure your Excel has the **Master** tab filled (company + phone fields).
