"""Convert final analysis prediction output into manuscript-ready primary statistics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from core import FACTORS, SCALES

def vector(value): return np.asarray(json.loads(value), dtype=float)
def pearson(x,y): return float(np.corrcoef(np.asarray(x,float),np.asarray(y,float))[0,1])
def bootstrap_pearson(x,y,seed):
    x=np.asarray(x,float); y=np.asarray(y,float); rng=np.random.default_rng(seed); results=[]
    for _ in range(20000):
        i=rng.integers(0,len(x),len(x)); value=pearson(x[i],y[i])
        if np.isfinite(value): results.append(value)
    return tuple(np.quantile(results,[.025,.975]))
def participant_vectors(predictions):
    fixed=predictions[predictions.record_type.eq("fixed") & np.isclose(predictions.scale,0)].copy(); records=[]
    for study_id,group in fixed.groupby("study_id",sort=True):
        records.append({"study_id":study_id,"true_y":np.vstack(group.true_y.map(vector)).mean(axis=0),"questionnaire_prediction":np.vstack(group.questionnaire_prediction.map(vector)).mean(axis=0),"predicted_visual_residual":np.vstack(group.predicted_visual_residual.map(vector)).mean(axis=0)})
    return pd.DataFrame(records)
def questionnaire_results(participants):
    y=np.vstack(participants.true_y); q=np.vstack(participants.questionnaire_prediction); rows=[]
    for j,factor in enumerate(FACTORS):
        lo,hi=bootstrap_pearson(y[:,j],q[:,j],1000+j); err=y[:,j]-q[:,j]
        rows.append({"factor":factor,"n_participants":len(y),"pearson_r":pearson(y[:,j],q[:,j]),"pearson_ci_low":lo,"pearson_ci_high":hi,"spearman_rho":float(spearmanr(y[:,j],q[:,j]).statistic),"mae":float(np.abs(err).mean()),"rmse":float(np.sqrt(np.mean(err**2))),"r2":float(1-np.sum(err**2)/np.sum((y[:,j]-y[:,j].mean())**2))})
    return pd.DataFrame(rows)
def visual_results(participants):
    y=np.vstack(participants.true_y); q=np.vstack(participants.questionnaire_prediction); visual=np.vstack(participants.predicted_visual_residual); residual=y-q; rows=[]
    for scope,index in [("Overall",None)]+[(factor,j) for j,factor in enumerate(FACTORS)]:
        truth=residual.ravel() if index is None else residual[:,index]; predicted=visual.ravel() if index is None else visual[:,index]; active=np.abs(truth)>=.25
        for predictor,value in (("Zero residual",np.zeros_like(truth)),("TinyCNN",predicted)):
            item={"scope":scope,"predictor":predictor,"n_observations":int(len(truth)),"residual_mae":float(np.abs(truth-value).mean()),"residual_mse":float(np.mean((truth-value)**2))}
            if predictor=="TinyCNN":
                positive=truth[active]>0; negative=truth[active]<0
                item.update({"direction_accuracy_abs_ge_025":float(np.mean(np.sign(value[active])==np.sign(truth[active]))),"balanced_accuracy_abs_ge_025":float((np.mean(value[active][positive]>0)+np.mean(value[active][negative]<0))/2),"pearson_r":pearson(truth,value),"spearman_rho":float(spearmanr(truth,value).statistic)})
            else: item.update({"direction_accuracy_abs_ge_025":np.nan,"balanced_accuracy_abs_ge_025":np.nan,"pearson_r":np.nan,"spearman_rho":np.nan})
            rows.append(item)
    return pd.DataFrame(rows)
def fixed_scale_results(predictions):
    fixed=predictions[predictions.record_type.eq("fixed")].groupby(["scale","repeat","fold"]).fusion_mae.mean().reset_index(name="mae"); zero=fixed[fixed.scale.eq(0)].set_index(["repeat","fold"]).mae; rows=[]
    for scale,group in fixed.groupby("scale"):
        values=group.mae.to_numpy(); rng=np.random.default_rng(8100+int(scale*1000)); boot=np.array([rng.choice(values,len(values),True).mean() for _ in range(10000)])
        rows.append({"scale":scale,"mean_mae":values.mean(),"sd_mae":values.std(ddof=1),"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),"folds_better_than_scale0":int(sum(group.set_index(["repeat","fold"]).mae<zero))})
    return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args(); predictions=pd.read_csv(a.predictions); output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    participants=participant_vectors(predictions); rq1=questionnaire_results(participants); rq2=visual_results(participants); fixed=fixed_scale_results(predictions); nested=predictions[predictions.record_type.eq("nested")].groupby(["repeat","fold"]).agg(selected_scale=("selected_scale","first"),outer_test_mae=("fusion_mae","mean"),questionnaire_mae=("questionnaire_mae","mean")).reset_index(); selected=nested.selected_scale.value_counts().reindex(SCALES,fill_value=0).rename_axis("scale").reset_index(name="selected_outer_folds")
    rq1.to_csv(output/"factor_questionnaire_results.csv",index=False);rq2.to_csv(output/"visual_residual_results.csv",index=False);fixed.to_csv(output/"fixed_scale_sensitivity.csv",index=False);selected.to_csv(output/"selected_scale_frequency.csv",index=False)
    overall=rq2[rq2.scope.eq("Overall")].set_index("predictor"); y=np.vstack(participants.true_y);q=np.vstack(participants.questionnaire_prediction);r=y-q;v=np.vstack(participants.predicted_visual_residual)
    summary={"n_participants":int(len(participants)),"questionnaire_participant_mae":float(np.abs(y-q).mean(axis=1).mean()),"zero_residual_mae":float(overall.loc["Zero residual","residual_mae"]),"tinycnn_residual_mae":float(overall.loc["TinyCNN","residual_mae"]),"tinycnn_minus_zero_participant_mae":float((np.abs(r-v).mean(axis=1)-np.abs(r).mean(axis=1)).mean()),"direction_accuracy_abs_ge_025":float(overall.loc["TinyCNN","direction_accuracy_abs_ge_025"]),"pearson_residual":float(overall.loc["TinyCNN","pearson_r"]),"spearman_residual":float(overall.loc["TinyCNN","spearman_rho"]),"scale_zero_mae":float(fixed.loc[np.isclose(fixed.scale,0),"mean_mae"].iloc[0]),"scale_005_mae":float(fixed.loc[np.isclose(fixed.scale,.05),"mean_mae"].iloc[0]),"nested_fusion_mae":float(nested.outer_test_mae.mean()),"alpha_zero_selected_folds":int(selected.loc[np.isclose(selected.scale,0),"selected_outer_folds"].iloc[0])}
    (output/"primary_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
if __name__=="__main__": main()
