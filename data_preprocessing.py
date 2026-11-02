from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def read_table(path):
    with open(path,encoding="utf-8-sig",errors="replace") as f:
        f.readline()
        header=f.readline()
        while header and not header.strip():
            header=f.readline()
    if header.split("\t",1)[0].strip().upper()=="TARGET":
        from src.run import load
        df=load(path)
        switches=[c for c in df if c.startswith("SWITCH_") and c.endswith("_STATE")]
        if not switches:raise ValueError(f"No switch-state fields in COSMOS telemetry {path}")
        df["__proxy_label__"]=df[switches].eq(1).any(axis=1).astype(int)
        return df.rename(columns={"t":"timestamp"})
    for sep in (None,r"\s+"):
        try:
            df=pd.read_csv(path,sep=sep,engine="python",comment="#")
            if len(df.columns)>1:return df
        except Exception:pass
    raise ValueError(f"Cannot read telemetry table {path}")

def prepare(folder):
    paths={k:[] for k in ("nominal","set1","set2")}
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in (".csv",".txt"):continue
        name=f.stem.lower()
        if "anomaly_set1" in name or "anomaly_set_1" in name:k="set1"
        elif "anomaly_set2" in name or "anomaly_set_2" in name:k="set2"
        elif "nominal" in name:k="nominal"
        else:continue
        paths[k].append(f)
    if not all(paths.values()):raise ValueError(f"Missing files: { {k:len(v) for k,v in paths.items()} }")
    frames={}
    for k,fs in paths.items():
        parts=[]
        for f in fs:
            df=read_table(f)
            label=next((c for c in df if str(c).lower().strip() in ("label","target","is_anomaly","anomaly_label","ground_truth","__proxy_label__")),None)
            if label is None and k!="nominal":raise ValueError(f"Missing row-level 0/1 labels in {f}; provide ground truth")
            if label=="__proxy_label__" and k=="nominal":
                df.pop(label)
                y=pd.Series(0,index=df.index)
            else:
                y=pd.to_numeric(df.pop(label),errors="coerce") if label else pd.Series(0,index=df.index)
            if y.isna().any() or not y.isin([0,1]).all():raise ValueError(f"Invalid labels in {f}")
            cols={}
            for c in df:
                if str(c).lower().strip() in ("time","timestamp","date","datetime","id","index","sequence"):continue
                a=pd.to_numeric(df[c],errors="coerce")
                if a.notna().mean()>=.8:cols[str(c)]=a
            if not cols:raise ValueError(f"No numeric columns in {f}")
            frame=pd.DataFrame(cols);frame["__label__"]=y.to_numpy(dtype=int);parts.append(frame)
        frames[k]=pd.concat(parts,ignore_index=True)
    features=sorted(set.intersection(*(set(df.columns)-{"__label__"} for df in frames.values())))
    if not features:raise ValueError("No common telemetry features")
    splits={}
    for k,df in frames.items():
        n=len(df);a=int(n*.6);b=int(n*.8)
        if min(a,b-a,n-b)<30:raise ValueError(f"{k}: insufficient rows")
        splits[k]={s:(df.iloc[lo:hi][features],df.iloc[lo:hi]["__label__"].to_numpy(dtype=int)) for s,lo,hi in (("train",0,a),("cal",a,b),("test",b,n))}
    normal=pd.concat([x.iloc[np.flatnonzero(y==0)] for k in splits for x,y in [splits[k]["train"]]],ignore_index=True)
    imputer=SimpleImputer(strategy="median");scaler=StandardScaler();scaler.fit(imputer.fit_transform(normal))
    return {k:{s:(scaler.transform(imputer.transform(x)).astype(np.float32),y) for s,(x,y) in group.items()} for k,group in splits.items()},features

def combined(data,split):
    return np.concatenate([data[k][split][0] for k in data]),np.concatenate([data[k][split][1] for k in data])

def normal_segments(data,split):
    # Only complete normal groups, no concatenation across anomaly gaps.
    segments=[]
    for k in data:
        x,y=data[k][split];starts=np.flatnonzero(np.diff(np.r_[1,y,1])==-1);ends=np.flatnonzero(np.diff(np.r_[1,y,1])==1)
        for a,b in zip(starts,ends):
            if b>a:segments.append(x[a:b])
    return segments
