Data files

1) lab_faq.csv (required)
- Columns: sku, kind, test_name, price_ZAR, turnaround_days, sample_prep, notes
- kind: "test" | "panel" | "address" | "" (legacy rows allowed)
- price_ZAR, turnaround_days: numeric or empty

2) sans_241_breakdown.csv (optional)
- Flexible schema; recommended columns:
  - panel or suite: grouping key, e.g., "SANS 241"
  - test_name: optional; overrides group to bind to a specific test
  - analyte: determinant name, e.g., "Arsenic (As)"
  - method: e.g., "ICP-MS", "AAS", "GC"
  - unit: e.g., "mg/L", "µg/L"
  - limit / mdl / dl: numeric limits if applicable
  - guideline / threshold: optional textual guidance
- If panel/suite/test_name are absent, rows are grouped under "SANS 241".
- All columns are preserved and emitted as-is in JSON.

Build
- Run: python scripts/build_catalog.py
- Output: data/compiled/lab_catalog.json
