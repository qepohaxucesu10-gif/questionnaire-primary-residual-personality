"""Exact portable 5x5 outer/5-fold-inner nested scale analysis."""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from core import DEVICE, SCALES, TinyVisualResidual, fit_questionnaire, fuse, seed, train_visual, visual_predictions

def oof_targets(train, image_dir, image_map, base):
    result = {}
    for n, (ii, jj) in enumerate(KFold(5, shuffle=True, random_state=base).split(train)):
        _, _, _, pred, _ = fit_questionnaire(train.iloc[ii].copy(), train.iloc[jj].copy(), image_dir, image_map, base + 10 + n)
        result.update({p: y - q for p, (y, q) in pred.items()})
        gc.collect(); torch.cuda.empty_cache()
    assert len(result) == len(train)
    return result

def fit_visual(train_data, test_data, targets, seed_value):
    seed(seed_value)
    model = train_visual(TinyVisualResidual().to(DEVICE), train_data, targets)
    result = visual_predictions(model, test_data)
    del model; gc.collect(); torch.cuda.empty_cache()
    return result

def main():
    p = argparse.ArgumentParser()
    for name in ("questionnaire-data", "image-dir", "image-map", "fold-assignments", "output-dir"):
        p.add_argument(f"--{name}", required=True)
    a = p.parse_args(); data = pd.read_csv(a.questionnaire_data); folds = pd.read_csv(a.fold_assignments); image_map = pd.read_csv(a.image_map)
    output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True); rows=[]; inner_tables=[]; selection_rows=[]
    for repeat, fold in sorted(folds[["repeat", "fold"]].drop_duplicates().itertuples(index=False)):
        split=folds[(folds.repeat==repeat)&(folds.fold==fold)]; train=data[data.study_id.isin(split[split.role=="train"].study_id)].copy(); test=data[data.study_id.isin(split[split.role=="test"].study_id)].copy(); inner=[]
        for inn,(ii,jj) in enumerate(KFold(5,shuffle=True,random_state=700000+repeat*100+fold).split(train)):
            it,iv=train.iloc[ii].copy(),train.iloc[jj].copy(); targets=oof_targets(it,a.image_dir,image_map,710000+repeat*100+fold*100+inn*10)
            dtr,dte,_,qvalid,_=fit_questionnaire(it,iv,a.image_dir,image_map,720000+repeat*100+fold*10+inn); visual=fit_visual(dtr,dte,targets,730000+repeat*100+fold*10+inn)
            for sid,(y,q) in qvalid.items():
                for scale in SCALES: inner.append({"scale":scale,"inner_fold":inn+1,"mae":float(np.abs(y-fuse(q,visual[sid],scale)).mean())})
        table=pd.DataFrame(inner); means=table.groupby("scale").mae.mean().reindex(SCALES); selected=float(means.index[np.argmin(means.to_numpy())]); inner_tables.append(table.assign(repeat=repeat,fold=fold,selected_scale=selected))
        targets=oof_targets(train,a.image_dir,image_map,740000+repeat*100+fold*100); dtr,dte,_,qtest,selected_items=fit_questionnaire(train,test,a.image_dir,image_map,750000+repeat*100+fold); selection_rows.append({"repeat":repeat,"fold":fold,"selected_item_count":len(selected_items),"item_ids":";".join(str(x+1) for x in selected_items)}); visual=fit_visual(dtr,dte,targets,760000+repeat*100+fold)
        for sid,(y,q) in qtest.items():
            residual=visual[sid]; base=float(np.abs(y-q).mean())
            for scale in SCALES:
                prediction=fuse(q,residual,scale); rows.append({"repeat":repeat,"fold":fold,"study_id":sid,"record_type":"fixed","scale":scale,"selected_scale":selected,"questionnaire_mae":base,"fusion_mae":float(np.abs(y-prediction).mean()),"true_y":json.dumps(y.tolist()),"questionnaire_prediction":json.dumps(q.tolist()),"predicted_visual_residual":json.dumps(residual.tolist()),"fusion_prediction":json.dumps(prediction.tolist())})
            prediction=fuse(q,residual,selected); rows.append({"repeat":repeat,"fold":fold,"study_id":sid,"record_type":"nested","scale":selected,"selected_scale":selected,"questionnaire_mae":base,"fusion_mae":float(np.abs(y-prediction).mean()),"true_y":json.dumps(y.tolist()),"questionnaire_prediction":json.dumps(q.tolist()),"predicted_visual_residual":json.dumps(residual.tolist()),"fusion_prediction":json.dumps(prediction.tolist())})
    predictions=pd.DataFrame(rows); predictions.to_csv(output/"participant_predictions_nested.csv",index=False); pd.concat(inner_tables,ignore_index=True).to_csv(output/"inner_scale_predictions.csv",index=False); pd.DataFrame(selection_rows).to_csv(output/"selected_items_by_fold.csv",index=False)
    nested=predictions[predictions.record_type=="nested"].groupby(["repeat","fold"]).agg(selected_scale=("selected_scale","first"),outer_test_mae=("fusion_mae","mean"),questionnaire_mae=("questionnaire_mae","mean")).reset_index(); nested.to_csv(output/"nested_scale_results.csv",index=False)
    fixed=predictions[predictions.record_type=="fixed"].groupby(["scale","repeat","fold"]).fusion_mae.mean().reset_index(name="mae"); zero=fixed[fixed.scale==0].set_index(["repeat","fold"]).mae; summary=[]
    for scale,group in fixed.groupby("scale"):
        x=group.mae.to_numpy(); rng=np.random.default_rng(8100+int(scale*1000)); boot=np.array([rng.choice(x,len(x),True).mean() for _ in range(10000)]); summary.append({"scale":scale,"mean_mae":x.mean(),"sd_mae":x.std(ddof=1),"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),"folds_better_than_scale0":int(sum(group.set_index(["repeat","fold"]).mae<zero))})
    pd.DataFrame(summary).to_csv(output/"fixed_scale_sensitivity.csv",index=False); nested.selected_scale.value_counts().reindex(SCALES,fill_value=0).rename_axis("scale").reset_index(name="selected_outer_folds").to_csv(output/"selected_scale_frequency.csv",index=False)

if __name__ == "__main__": main()
