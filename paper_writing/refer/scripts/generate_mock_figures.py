"""Generate unified MOCK -- NOT EMPIRICAL CLAIMS CSVs and publication-layout figures."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FIG = ROOT / "figures_mock"
EXP.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

def write(name, rows):
    with (EXP / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

datasets = ["NYC", "TKY", "CA"]
write("mock_dataset_stats.csv", [
 {"dataset":"NYC","users":"[TBD]","pois":"[TBD]","checkins":"[TBD]","train":3870,"validation":"[TBD]","test":988,"history":"[TBD]","long_tail":"[TBD]","geosemid_samples":"[TBD]","teacher_samples":"[TBD]","ranker_coverage":.688,"router_coverage":.746,"status":"MOCK -- NOT EMPIRICAL CLAIMS"},
 {"dataset":"TKY","users":"[TBD]","pois":"[TBD]","checkins":"[TBD]","train":11850,"validation":"[TBD]","test":2206,"history":"[TBD]","long_tail":"[TBD]","geosemid_samples":"[TBD]","teacher_samples":"[TBD]","ranker_coverage":.671,"router_coverage":.732,"status":"MOCK -- NOT EMPIRICAL CLAIMS"},
 {"dataset":"CA","users":"[TBD]","pois":"[TBD]","checkins":"[TBD]","train":6616,"validation":"[TBD]","test":1818,"history":"[TBD]","long_tail":"[TBD]","geosemid_samples":"[TBD]","teacher_samples":"[TBD]","ranker_coverage":.641,"router_coverage":.711,"status":"MOCK -- NOT EMPIRICAL CLAIMS"}])

methods=["FPMC","ST-RNN","DeepMove","GETNext","CoMaPOI","Semantic","SingleRetriever","Uniform","RRF","Full"]
rows=[]
for d,base in zip(datasets,[.40,.385,.344]):
 for i,m in enumerate(methods):
  hr=base-(9-i)*.008 if m!="Full" else base
  rows.append({"dataset":d,"method":m,"hr1":round(hr*.45,3),"hr5":round(hr*.82,3),"hr10":round(hr,3),"ndcg5":round(hr*.65,3),"ndcg10":round(hr*.72,3),"mrr":round(hr*.57,3),"std":".004","status":"MOCK -- NOT EMPIRICAL CLAIMS"})
write("mock_overall_results.csv",rows)

cov=[]
for d,full in zip(datasets,[.688,.671,.641]):
 for m,off in [("Spatial",.12),("Transition",.10),("Temporal",.16),("Semantic",.09),("Union",.025),("Uniform",.015),("Global",.01),("RRF",.013),("Router",0)]:
  r50=full-off; cov.append({"dataset":d,"method":m,"r10":round(r50-.23,3),"r25":round(r50-.105,3),"r50":round(r50,3),"median_rank":round(18+off*100,1),"candidate_size":25,"router_coverage":round(.74-off/2,3),"miss":round(1-r50,3),"rank_error":round((1-r50)*.55,3),"correct":round(1-(1-r50)*1.55,3),"status":"MOCK -- NOT EMPIRICAL CLAIMS"})
write("mock_candidate_results.csv",cov)

abl=["Atomic ID","Metadata","Continuous embedding","Free-text","Random code","Unfine-tuned","Semantic only","Geo only","Category only","Geo+Category","Geo+Semantic","Category+Semantic","Full","Full w/o backoff"]
write("mock_geosemid_ablation.csv",[{"variant":x,"recall25":round(.42+i*.012,3),"hr10":round(.26+i*.007,3),"ndcg10":round(.18+i*.006,3),"longtail_hr10":round(.16+i*.006,3),"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for i,x in enumerate(abl)])
router=["Union","Uniform","Fixed","Global","RRF","Unsupervised","Binary-hit","Rank-utility","No GeoSemID","No mobility","No temporal","Full"]
write("mock_router_ablation.csv",[{"variant":x,"recall25":round(.50+i*.006,3),"hr10":round(.30+i*.004,3),"entropy":round(1.28-i*.018,3),"effective_experts":round(3.6-i*.04,2),"worst_load":.12,"max_load":round(.36+i*.008,3),"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for i,x in enumerate(router)])
write("mock_evidence_ablation.csv",[{"variant":x,"conditional_hr10":round(.52+i*.012,3),"end_hr10":round(.31+i*.008,3),"ndcg10":round(.22+i*.007,3),"invalid_rate":round(.06-i*.006,3),"latency_ms":round(80+i*8,1),"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for i,x in enumerate(["No evidence","Membership","Percentile","Weights","Full","Shuffled","Random","Open generation","Candidate likelihood","No reverse data","Reverse data","Teacher reference","Full local"])])
write("mock_expert_combinations.csv",[{"slice":x,"recall25":round(.46+i*.011,3),"hr10":round(.27+i*.009,3),"ndcg10":round(.19+i*.007,3),"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for i,x in enumerate(["Spatial","Transition","Temporal","Semantic","Spatial+Transition","Transition+Temporal","Spatial+Semantic","All four","All four+Router","Head","Mid","Long-tail","Short","Sparse","Weekend","Evening"])])
write("mock_efficiency.csv",[{"stage":x,"time":"[TBD]","latency_ms":v,"memory_gb":"[TBD]","api_calls":"[TBD]","status":"MOCK -- NOT EMPIRICAL CLAIMS"} for x,v in [("GeoSemID offline",0),("Teacher reverse data",0),("Expert retrieval",18),("Router",1),("Fusion",1),("Local ranking",72)]])
load=np.array([[.31,.28,.19,.22],[.29,.30,.18,.23],[.35,.24,.17,.24]])
write("mock_router_load.csv",[{"dataset":d,"spatial":a,"transition":b,"temporal":c,"semantic":e,"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for d,(a,b,c,e) in zip(datasets,load)])
write("mock_router_entropy.csv",[{"dataset":d,"entropy":round(1.25-i*.04,3),"top1_hit":round(.51+i*.03,3),"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for i,d in enumerate(datasets)])
write("mock_expert_overlap.csv",[{"pattern":p,"share":v,"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for p,v in [("S only",.09),("T only",.08),("P only",.06),("M only",.11),("multi-hit",.48),("all-miss",.18)]])
write("mock_error_decomposition.csv",[{"dataset":d,"method":m,"retrieval_miss":a,"ranking_error":b,"correct":round(1-a-b,3),"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for d in datasets for m,a,b in [("Semantic",.40,.36),("Uniform",.34,.34),("Router",.29,.33),("Full",.27,.30)]])
write("mock_geosemid_quality.csv",[{"metric":x,"value":v,"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for x,v in [("Validity",.972),("UNK",.018),("Collision",.214),("POIs/code",3.8),("Intra-code distance",1.7),("Coherence",.76)]])
write("mock_teacher_student.csv",[{"variant":x,"valid":v,"consistency":c,"hr10":h,"latency_ms":l,"status":"MOCK -- NOT EMPIRICAL CLAIMS"} for x,v,c,h,l in [("Local labels",.93,.89,.31,76),("Reverse teacher",.96,.94,.34,76),("Reverse+evidence",.97,.95,.36,82),("Teacher reference",.99,.98,.39,840)]])

plt.style.use("seaborn-v0_8-whitegrid")
def save(name):
 plt.tight_layout(); plt.savefig(FIG/name,dpi=220); plt.savefig(FIG/name); plt.close()
fig,ax=plt.subplots(1,2,figsize=(9,3)); bottom=np.zeros(3)
for i,label in enumerate(["Spatial","Transition","Temporal","Semantic"]):
 ax[0].bar(datasets,load[:,i],bottom=bottom,label=label); bottom+=load[:,i]
ax[0].legend(fontsize=7); ax[0].set_title("Router load (Mock)"); ax[1].boxplot([[1.1,1.2,1.3],[1.0,1.15,1.27],[.98,1.1,1.2]],labels=datasets); ax[1].set_title("Entropy (Mock)"); save(Path("router_load.pdf"))
fig,ax=plt.subplots(1,2,figsize=(9,3)); ax[0].bar(["S","T","P","M","Multi","Miss"],[.09,.08,.06,.11,.48,.18]); ax[0].set_title("Target-hit overlap"); im=ax[1].imshow([[1,.25,.18,.31],[.25,1,.22,.19],[.18,.22,1,.16],[.31,.19,.16,1]],vmin=0,vmax=1); ax[1].set_title("Candidate Jaccard"); fig.colorbar(im,ax=ax[1]); save(Path("expert_overlap.pdf"))
fig,ax=plt.subplots(figsize=(6,3)); x=np.arange(4); 
for m,a,b in [("Semantic",.40,.36),("Uniform",.34,.34),("Router",.29,.33),("Full",.27,.30)]: ax.bar(m,a,label="miss" if m=="Semantic" else ""); ax.bar(m,b,bottom=a,label="rank error" if m=="Semantic" else ""); ax.bar(m,1-a-b,bottom=a+b,label="correct" if m=="Semantic" else "")
ax.legend(fontsize=7); ax.set_title("Error decomposition (Mock)"); save(Path("error_decomposition.pdf"))
fig,ax=plt.subplots(figsize=(6,3)); k=[10,25,50]; ax.plot(k,[.41,.54,.64],marker="o",label="Recall"); ax.plot(k,[32,49,83],marker="s",label="Latency ms"); ax.legend(); ax.set_title("Recall--latency budget (Mock)"); save(Path("recall_latency.pdf"))
fig,ax=plt.subplots(1,2,figsize=(9,3)); ax[0].bar(["Valid","UNK","Collision"],[.972,.018,.214]); ax[1].boxplot([[1.3,1.6,1.8,2.0],[.69,.74,.78,.82]],labels=["Distance","Coherence"]); save(Path("geosemid_quality.pdf"))
fig,ax=plt.subplots(figsize=(6,3)); conf=[.2,.4,.6,.8]; ax.plot(conf,[.31,.43,.56,.68],marker="o"); ax.plot(conf,conf,"--"); ax.set_title("Router calibration (Mock)"); ax.set_xlabel("Router confidence"); ax.set_ylabel("Target-hit rate"); save(Path("router_calibration.pdf"))
fig,ax=plt.subplots(figsize=(7,3)); ax.axis("off"); ax.table(cellText=[["Commute","Spatial .42","Target rank 2"],["Dining","Temporal .38","Target rank 3"],["Explore","Semantic .44","Target rank 4"]],colLabels=["Case","Top evidence","Fused outcome"],loc="center"); ax.set_title("Qualitative cases (Mock)"); save(Path("case_studies.pdf"))
