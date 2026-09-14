"""Teacher-salience external analysis for the final analysis configuration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from core import FACTORS, teacher_top3

N_DRAWS = 20000

def vector(value): return np.asarray(json.loads(value), dtype=float)
def score(target, predicted):
    overlap=len(target & predicted)
    return overlap, int(overlap >= 1), int(overlap == 3), overlap / (6 - overlap), overlap / 3
def bootstrap(values, seed):
    x=np.asarray(values, dtype=float); rng=np.random.default_rng(seed)
    draws=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(N_DRAWS)])
    return tuple(np.quantile(draws,[.025,.975]))
def paired_permutation(values, seed):
    x=np.asarray(values,dtype=float); rng=np.random.default_rng(seed); observed=x.mean()
    draws=np.array([(x*rng.choice([-1,1],len(x))).mean() for _ in range(N_DRAWS)])
    return float((np.sum(np.abs(draws)>=abs(observed))+1)/(N_DRAWS+1))
def sources_from_predictions(predictions):
    frames=[
        ("T0_full_form_reference","true_y",predictions[predictions.record_type.eq("nested")]),
        ("T1_questionnaire_heldout","questionnaire_prediction",predictions[predictions.record_type.eq("nested")]),
        ("T2_fixed_scale_005_fusion","fusion_prediction",predictions[predictions.record_type.eq("fixed") & np.isclose(predictions.scale,.05)]),
        ("T3_nested_selected_fusion","fusion_prediction",predictions[predictions.record_type.eq("nested")]),
    ]
    return {name:{str(sid):np.vstack(group[column].map(vector)).mean(axis=0) for sid,group in frame.groupby("study_id")} for name,column,frame in frames}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--teacher-data",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
    predictions=pd.read_csv(a.predictions); teacher=pd.read_csv(a.teacher_data); required={"study_id","teacher_trait_set"}; assert required.issubset(teacher.columns)
    methods=sources_from_predictions(predictions); rows=[]
    for record in teacher.itertuples(index=False):
        target={FACTORS.index(x) for x in record.teacher_trait_set.split(";")}
        for method,values in methods.items():
            selected=teacher_top3(values[str(record.study_id)]); overlap,any_hit,exact,jaccard,precision=score(target,selected)
            rows.append({"study_id":str(record.study_id),"method":method,"teacher_trait_set":record.teacher_trait_set,"predicted_salient_set":";".join(FACTORS[i] for i in sorted(selected)),"overlap_count":overlap,"any_hit":any_hit,"exact_match":exact,"jaccard":jaccard,"precision_at_3":precision,"recall_at_3":precision})
    alignment=pd.DataFrame(rows); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); alignment.to_csv(out/"teacher_alignment_participant.csv",index=False)
    teacher_sets=[{FACTORS.index(x) for x in value.split(";")} for value in teacher.teacher_trait_set]
    rng=np.random.default_rng(9600); chance=[]
    for _ in range(N_DRAWS): chance.append([score(target,set(rng.choice(16,3,replace=False))) for target in teacher_sets])
    chance=np.asarray(chance,dtype=float).mean(axis=1)
    summary=[]; chance_rows=[]
    for index,(method,group) in enumerate(alignment.groupby("method",sort=True)):
        observed={"overlap_count":group.overlap_count.mean(),"any_hit":group.any_hit.mean(),"exact_match":group.exact_match.mean(),"jaccard":group.jaccard.mean()}
        lo,hi=bootstrap(group.overlap_count,9700+index)
        summary.append({"method":method,"n_participants":len(group),**observed,"precision_at_3":group.precision_at_3.mean(),"overlap_ci_low":lo,"overlap_ci_high":hi})
        for j,metric in enumerate(("overlap_count","any_hit","exact_match","jaccard")):
            chance_rows.append({"method":method,"metric":metric,"observed_mean":observed[metric],"chance_mean":chance[:,j].mean(),"chance_ci_low":np.quantile(chance[:,j],.025),"chance_ci_high":np.quantile(chance[:,j],.975),"monte_carlo_p_greater_or_equal":(np.sum(chance[:,j]>=observed[metric])+1)/(N_DRAWS+1)})
    pd.DataFrame(summary).to_csv(out/"teacher_alignment_summary.csv",index=False); pd.DataFrame(chance_rows).to_csv(out/"teacher_chance_baseline.csv",index=False)
    comparisons=[]; questionnaire=alignment[alignment.method.eq("T1_questionnaire_heldout")].set_index("study_id")
    for offset,method in enumerate(("T2_fixed_scale_005_fusion","T3_nested_selected_fusion")):
        fusion=alignment[alignment.method.eq(method)].set_index("study_id").loc[questionnaire.index]
        for metric in ("overlap_count","jaccard"):
            difference=fusion[metric]-questionnaire[metric]; lo,hi=bootstrap(difference,9800+len(comparisons)); w=float(wilcoxon(difference).pvalue) if np.any(difference!=0) else 1.0
            comparisons.append({"comparison":f"{method} - T1_questionnaire_heldout","metric":metric,"mean_difference":difference.mean(),"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"paired_permutation_p":paired_permutation(difference,9900+len(comparisons)),"wilcoxon_p":w,"paired_d":difference.mean()/difference.std(ddof=1) if difference.std(ddof=1)>0 else 0.0,"mcnemar_exact_p":np.nan})
        before=questionnaire.any_hit.to_numpy(); after=fusion.any_hit.to_numpy(); difference=after-before; discordant=before!=after; lo,hi=bootstrap(difference,9800+len(comparisons))
        comparisons.append({"comparison":f"{method} - T1_questionnaire_heldout","metric":"any_hit","mean_difference":difference.mean(),"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"paired_permutation_p":paired_permutation(difference,9900+len(comparisons)),"wilcoxon_p":np.nan,"paired_d":np.nan,"mcnemar_exact_p":binomtest(int(np.sum((before==0)&(after==1))),int(discordant),.5).pvalue if discordant.any() else 1.0})
    pd.DataFrame(comparisons).to_csv(out/"teacher_paired_comparisons.csv",index=False)
if __name__=="__main__": main()
