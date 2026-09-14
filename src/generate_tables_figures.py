from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser();p.add_argument("--derived-results",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();source=Path(a.derived_results);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rq1=pd.read_csv(source/"factor_questionnaire_results.csv");fig,ax=plt.subplots(figsize=(7.09,4.77));ax.errorbar(rq1.pearson_r,rq1.factor,xerr=[rq1.pearson_r-rq1.pearson_ci_low,rq1.pearson_ci_high-rq1.pearson_r],fmt="o");ax.set(xlabel="Pearson r",ylabel="Factor");fig.tight_layout();fig.savefig(out/"Figure_2.tif",dpi=300);plt.close(fig)
    scale=pd.read_csv(source/"fixed_scale_sensitivity.csv");frequency=pd.read_csv(source/"selected_scale_frequency.csv");fig,ax=plt.subplots(1,2,figsize=(7.09,3.55));ax[0].errorbar(scale.scale,scale.mean_mae,yerr=scale.sd_mae,fmt="o-");ax[0].set(xlabel="Scale",ylabel="Mean outer-fold MAE");ax[1].bar(frequency.scale.astype(str),frequency.selected_outer_folds);ax[1].set(xlabel="Scale",ylabel="Selected outer folds");fig.tight_layout();fig.savefig(out/"Figure_3.tif",dpi=300);plt.close(fig)
if __name__=="__main__":main()
