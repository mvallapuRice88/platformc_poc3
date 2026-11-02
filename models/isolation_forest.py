import numpy as np
from sklearn.ensemble import IsolationForest

def train(segments,cal_segments,fpr,seed):
    model=IsolationForest(n_estimators=300,random_state=seed,n_jobs=-1,contamination="auto")
    model.fit(np.concatenate(segments))
    scores=np.concatenate([-model.score_samples(s) for s in cal_segments if len(s)])
    return model,float(np.quantile(scores,1-fpr))

def predict(model,threshold,x):return (-model.score_samples(x)>threshold).astype(int)
