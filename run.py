import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_preprocessing import prepare,combined,normal_segments
from utils import metrics
from models import isolation_forest,gradient_boost,xgboost_model,lstm,msgmamba

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data");p.add_argument("--output-dir",default="outputs")
    p.add_argument("--epochs",type=int,default=10);p.add_argument("--window",type=int,default=24)
    p.add_argument("--fpr",type=float,default=.01);p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if not 0<args.fpr<1:raise ValueError("fpr must be between 0 and 1")
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    data,features=prepare(args.data_dir);device="cuda" if torch.cuda.is_available() else "cpu"
    train_x,train_y=combined(data,"train")
    if len(np.unique(train_y))<2:raise ValueError("Labeled training split must contain both normal and anomaly samples")
    normal_train=normal_segments(data,"train");normal_cal=normal_segments(data,"cal")
    if not normal_train or not normal_cal:raise ValueError("Need normal train/calibration segments")
    models={}
    print("Training Isolation Forest")
    models["isolation_forest"]=(*isolation_forest.train(normal_train,normal_cal,args.fpr,args.seed),isolation_forest.predict)
    print("Training Gradient Boosting")
    models["gradient_boost"]=(gradient_boost.train(train_x,train_y,args.seed),None,gradient_boost.predict)
    print("Training XGBoost")
    models["xgboost"]=(xgboost_model.train(train_x,train_y,args.seed),None,xgboost_model.predict)
    print("Training LSTM")
    models["lstm"]=(*lstm.train(normal_train,normal_cal,len(features),args.window,args.epochs,args.fpr,device),lstm.predict)
    print("Training MSGMamba-inspired baseline")
    models["msgmamba_inspired"]=(*msgmamba.train(normal_train,normal_cal,len(features),args.window,args.epochs,args.fpr,device),msgmamba.predict)
    rows=[]
    for name,(model,threshold,predict) in models.items():
        for group in ("set1","set2","nominal"):
            x,y=data[group]["test"]
            if name in ("lstm","msgmamba_inspired"):
                valid,pred=predict(model,threshold,x,args.window,device)
                actual=y[valid];excluded=int((~valid).sum())
            elif threshold is not None:
                pred=predict(model,threshold,x);actual=y;excluded=0
            else:
                pred=predict(model,x);actual=y;excluded=0
            if not len(actual):raise ValueError(f"{group} has no evaluable rows for {name}")
            rows.append(dict(model=name,run="nominal_holdout" if group=="nominal" else group,**metrics(actual,pred),excluded_warmup_samples=excluded,test_samples=len(actual)))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    report=pd.DataFrame(rows);report.to_csv(out/"model_metrics.csv",index=False)
    print(report.to_string(index=False));print(f"Saved {out/'model_metrics.csv'}")
    print("WARNING: MSGMamba-inspired is a GRU-based approximation, not the published architecture.")
    print("WARNING: Chronological splits can still leak anomaly episodes across boundaries; use event-grouped splits when episode IDs exist.")

if __name__=="__main__":main()
