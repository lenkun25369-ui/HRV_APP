# parse_fhir_ecg_to_csv.py
import json, sys
import pandas as pd

FHIR_JSON = sys.argv[1]
OUT_CSV = sys.argv[2] if len(sys.argv) >= 3 else "ECG_5min.csv"

with open(FHIR_JSON, "r") as f:
    obs = json.load(f)

sampled = obs["valueSampledData"]
ecg_values = list(map(float, sampled["data"].split()))

df = pd.DataFrame({"II": ecg_values})
df.to_csv(OUT_CSV, index=False)

print(f"[OK] ECG saved to {OUT_CSV}, n={len(df)}")
