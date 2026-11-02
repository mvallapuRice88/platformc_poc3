import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),precision=float(precision),recall=float(recall),f1=float(f1))

def windows(x,w):
    if len(x)<=w:return np.empty((0,w,x.shape[1]),np.float32),np.empty((0,x.shape[1]),np.float32)
    return np.stack([x[i-w:i] for i in range(w,len(x))]).astype(np.float32),x[w:].astype(np.float32)

def score_sequence(model,x,w,device):
    xx,yy=windows(x,w); scores=np.full(len(x),np.nan)
    if not len(xx):return scores
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(xx),512):out.append(model(torch.tensor(xx[i:i+512],device=device)).cpu().numpy())
    scores[w:]=np.mean((np.concatenate(out)-yy)**2,axis=1)
    return scores

def train_sequence(model,segments,cal_segments,w,epochs,fpr,device):
    parts=[windows(s,w) for s in segments if len(s)>w]
    if not parts:raise ValueError("No normal training sequences longer than window")
    xx=np.concatenate([a for a,b in parts]);yy=np.concatenate([b for a,b in parts])
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(xx),torch.tensor(yy)),batch_size=128,shuffle=True)
    model.to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    for ep in range(epochs):
        model.train();losses=[]
        for a,b in loader:
            a,b=a.to(device),b.to(device);opt.zero_grad();loss=torch.nn.functional.mse_loss(model(a),b);loss.backward();opt.step();losses.append(loss.item())
        print(f"epoch {ep+1}/{epochs}: {np.mean(losses):.6f}")
    cal=np.concatenate([score_sequence(model,s,w,device) for s in cal_segments if len(s)>w]);cal=cal[np.isfinite(cal)]
    if not len(cal):raise ValueError("No nominal calibration sequences")
    return model,float(np.quantile(cal,1-fpr))
