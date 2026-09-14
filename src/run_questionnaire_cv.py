from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from core import FACTORS,fit_questionnaire
def main():
    p=argparse.ArgumentParser();p.add_argument("--questionnaire-data",required=True);p.add_argument("--image-dir",required=True);p.add_argument("--image-map",required=True);p.add_argument("--fold-assignments",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args()
    data=pd.read_csv(a.questionnaire_data);folds=pd.read_csv(a.fold_assignments);image_map=pd.read_csv(a.image_map);rows=[]
    for repeat,fold in sorted(folds[["repeat","fold"]].drop_duplicates().itertuples(index=False)):
        split=folds[(folds.repeat==repeat)&(folds.fold==fold)];train=data[data.study_id.isin(split[split.role=="train"].study_id)].copy();test=data[data.study_id.isin(split[split.role=="test"].study_id)].copy();_,_,_,pred,selected=fit_questionnaire(train,test,a.image_dir,image_map,750000+repeat*100+fold)
        for study_id,(y,q) in pred.items(): rows.append({"study_id":study_id,"repeat":repeat,"fold":fold,"true_y":json.dumps(y.tolist()),"questionnaire_prediction":json.dumps(q.tolist()),"selected_item_count":len(selected)})
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/"questionnaire_predictions.csv",index=False)
if __name__=="__main__":main()
