from pathlib import Path
import csv, sys
root=Path(__file__).resolve().parents[1]/"experiments"
def rows(n): return list(csv.DictReader((root/n).open(encoding="utf-8")))
errors=[]
for r in rows("mock_candidate_results.csv"):
    a,b,c=map(float,(r["r10"],r["r25"],r["r50"]))
    if not a<=b<=c: errors.append("nonmonotone recall")
    if abs(float(r["miss"])+float(r["rank_error"])+float(r["correct"])-1)>1e-6: errors.append("error sum")
for r in rows("mock_overall_results.csv"):
    if float(r["ndcg10"])>float(r["hr10"]): errors.append("ndcg exceeds hr")
for r in rows("mock_router_load.csv"):
    if abs(sum(float(r[x]) for x in ["spatial","transition","temporal","semantic"])-1)>1e-6: errors.append("router sum")
for f in root.glob("mock_*.csv"):
    if f.name == "mock_results.csv": continue
    if "MOCK -- NOT EMPIRICAL CLAIMS" not in f.read_text(encoding="utf-8"): errors.append(f"missing label {f.name}")
if errors: print("FAILED",sorted(set(errors))); sys.exit(1)
print("PASSED: Mock consistency checks")
