"""NOT STUDY DATA. Exercises public primary-summary and teacher-analysis scripts."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/"src"; FACTORS=["A","B","C","E","F","G","H","I","L","M","N","O","Q1","Q2","Q3","Q4"]; SCALES=[0,.005,.01,.025,.05,.075,.1]
def values(offset): return (np.arange(16,dtype=float)%10+1+offset).clip(1,10)
def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp); rows=[]
        for repeat in range(1,6):
            for fold in range(1,6):
                sid=f"S{fold:02d}"; truth=np.random.default_rng(fold).uniform(1.5,9.5,16); target_residual=np.random.default_rng(100+fold).uniform(-.5,.5,16); question=np.clip(truth-target_residual,1,10); residual=target_residual+np.random.default_rng(200+fold).normal(0,.03,16)
                for scale in SCALES:
                    pred=np.clip(question+scale*residual,1,10); rows.append({"study_id":sid,"repeat":repeat,"fold":fold,"record_type":"fixed","scale":scale,"selected_scale":0,"questionnaire_mae":float(np.abs(truth-question).mean()),"fusion_mae":float(np.abs(truth-pred).mean()),"true_y":json.dumps(truth.tolist()),"questionnaire_prediction":json.dumps(question.tolist()),"predicted_visual_residual":json.dumps(residual.tolist()),"fusion_prediction":json.dumps(pred.tolist())})
                rows.append({"study_id":sid,"repeat":repeat,"fold":fold,"record_type":"nested","scale":0,"selected_scale":0,"questionnaire_mae":float(np.abs(truth-question).mean()),"fusion_mae":float(np.abs(truth-question).mean()),"true_y":json.dumps(truth.tolist()),"questionnaire_prediction":json.dumps(question.tolist()),"predicted_visual_residual":json.dumps(residual.tolist()),"fusion_prediction":json.dumps(question.tolist())})
        predictions=root/"predictions.csv"; pd.DataFrame(rows).to_csv(predictions,index=False); criterion=root/"teacher.csv"; pd.DataFrame({"study_id":[f"S{i:02d}" for i in range(1,6)],"teacher_trait_set":["A;B;C"]*5}).to_csv(criterion,index=False)
        summary=root/"summary"; teacher=root/"teacher"; subprocess.run([sys.executable,str(SRC/"summarize_primary_results.py"),"--predictions",str(predictions),"--output-dir",str(summary)],check=True); subprocess.run([sys.executable,str(SRC/"run_teacher_external_analysis.py"),"--predictions",str(predictions),"--teacher-data",str(criterion),"--output-dir",str(teacher)],check=True)
        required=[summary/"factor_questionnaire_results.csv",summary/"visual_residual_results.csv",summary/"fixed_scale_sensitivity.csv",summary/"selected_scale_frequency.csv",summary/"primary_summary.json",teacher/"teacher_alignment_participant.csv",teacher/"teacher_alignment_summary.csv",teacher/"teacher_chance_baseline.csv",teacher/"teacher_paired_comparisons.csv"]
        assert all(x.exists() for x in required)
    print("Synthetic public analysis smoke test passed. NOT STUDY DATA.")
if __name__=="__main__":main()
