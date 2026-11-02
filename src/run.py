"""Platform C PoC3 offline benchmark. No spacecraft command/mitigation capability."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

FIELDS={
 'GENERIC_EPS':['BATT_VOLTAGE','BATT_CURRENT','LOAD_BUS_CURRENT','BUS_3P3_CURRENT','BUS_12_CURRENT','LOAD_POWER']+[f'SWITCH_{i}_STATE' for i in range(8)],
 'GENERIC_MAG':['RAW_MAG_X','RAW_MAG_Y','RAW_MAG_Z'],
 'GENERIC_ADCS':['BVB_X','BVB_Y','BVB_Z','MCMD_X','MCMD_Y','MCMD_Z'],
 'GENERIC_TORQUER':['TORQUER_PERCENT_ON_0','TORQUER_PERCENT_ON_1','TORQUER_PERCENT_ON_2']}
BLOCK={'GENERIC_EPS':0,'GENERIC_ADCS':1,'GENERIC_IMU':2,'GENERIC_MAG':3,'GENERIC_REACTION_WHEEL':4,'GENERIC_TORQUER':5}

def load(path,start=320):
 """Read COSMOS TSV with a source-path first line and repeated packet-block headers."""
 streams={k:[] for k in FIELDS}
 with open(path,encoding='utf-8-sig',errors='replace') as f:
  f.readline()
  header_line=f.readline()
  while header_line and not header_line.strip():
   header_line=f.readline()
  header=header_line.rstrip('\n').split('\t')
  starts=[i for i,x in enumerate(header) if x=='PACKET_TIMESECONDS']
  positions={}
  for target,fields in FIELDS.items():
   b=BLOCK[target]
   if b>=len(starts):continue
   a=starts[b];z=starts[b+1] if b+1<len(starts) else len(header)
   positions[target]={key:next((i for i in range(a,z) if header[i]==key),None) for key in ['CCSDS_SECONDS']+fields}
  for line in f:
   cells=line.rstrip('\r\n').split('\t');target=cells[0]
   if target not in positions:continue
   def get(key):
    i=positions[target].get(key)
    return cells[i].strip() if i is not None and i<len(cells) else ''
   try:t=float(get('CCSDS_SECONDS'))
   except ValueError:continue
   if t<start:continue
   row={'t':t}
   for key in FIELDS[target]:
    v=get(key)
    if key.startswith('SWITCH_'):
     row[key]=1. if v.upper()=='ON' else (0. if v.upper()=='OFF' else np.nan)
    else:row[key]=pd.to_numeric(v,errors='coerce')
   streams[target].append(row)
 streams={k:pd.DataFrame(v).sort_values('t').drop_duplicates('t',keep='last') for k,v in streams.items() if v}
 if not streams:raise ValueError('No recognized telemetry after startup cutoff')
 base='GENERIC_MAG' if 'GENERIC_MAG' in streams else max(streams,key=lambda k:len(streams[k]))
 df=streams[base]
 for key,part in streams.items():
  if key!=base:df=pd.merge_asof(df,part,on='t',direction='nearest',tolerance=3)
 return df.reset_index(drop=True)

def prep(train,other):
 columns=[c for c in train if c!='t' and pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True)>1 and all(c in d for d in other)]
 imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(train[columns]);sc=RobustScaler().fit(imp.transform(train[columns]))
 return columns,lambda d:np.clip(sc.transform(imp.transform(d[columns])),-30,30)

def score_ml(name,train,test):
 if name=='robust_z':
  med=np.median(train,axis=0);mad=np.maximum(1.4826*np.median(abs(train-med),axis=0),.05)
  return lambda x:np.max(abs((x-med)/mad),axis=1)
 if name=='isolation_forest':
  model=IsolationForest(n_estimators=200,random_state=42,n_jobs=-1).fit(train)
  return lambda x:-model.score_samples(x)
 model=PCA(n_components=max(1,min(train.shape[1]-1,int(.8*train.shape[1]))),random_state=42).fit(train)
 return lambda x:np.mean((x-model.inverse_transform(model.transform(x)))**2,axis=1)

def score_deep(name,train,epochs):
 import torch
 from torch import nn
 torch.manual_seed(42);d=train.shape[1]
 if name=='autoencoder':net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,16),nn.ReLU(),nn.Linear(16,64),nn.ReLU(),nn.Linear(64,d))
 else:
  class LSTM(nn.Module):
   def __init__(self):super().__init__();self.rnn=nn.LSTM(d,32,batch_first=True);self.out=nn.Linear(32,d)
   def forward(self,x):return self.out(self.rnn(x)[0])
  net=LSTM()
 def windows(x):
  if len(x)<16:raise ValueError('At least 16 samples required')
  return np.stack([x[i:i+16] for i in range(len(x)-15)]).astype('float32')
 data=windows(train) if name=='lstm_autoencoder' else train.astype('float32')
 loader=torch.utils.data.DataLoader(torch.from_numpy(data),batch_size=256,shuffle=True)
 opt=torch.optim.Adam(net.parameters(),lr=.001)
 net.train()
 for _ in range(epochs):
  for batch in loader:
   opt.zero_grad();loss=((net(batch)-batch)**2).mean();loss.backward();opt.step()
 net.eval()
 def score(x):
  arr=windows(x) if name=='lstm_autoencoder' else x.astype('float32')
  with torch.no_grad():
   vals=[]
   for part in np.array_split(arr,max(1,int(np.ceil(len(arr)/512)))):
    t=torch.from_numpy(part);pred=net(t)
    if name=='lstm_autoencoder':vals.extend(((pred[:,-1]-t[:,-1])**2).mean(1).numpy())
    else:vals.extend(((pred-t)**2).mean(1).numpy())
  vals=np.asarray(vals)
  return np.pad(vals,(15,0),mode='edge') if name=='lstm_autoencoder' else vals
 return score

def main():
 p=argparse.ArgumentParser();p.add_argument('--nominal',required=True);p.add_argument('--set1',required=True);p.add_argument('--set2',required=True)
 p.add_argument('--out',default='outputs');p.add_argument('--start',type=float,default=320);p.add_argument('--deep',action='store_true');p.add_argument('--epochs',type=int,default=12)
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 data={name:load(path,a.start) for name,path in [('nominal',a.nominal),('set1',a.set1),('set2',a.set2)]}
 nominal=data['nominal'];cut=int(.7*len(nominal));train=nominal.iloc[:cut];hold=nominal.iloc[cut:]
 if len(train)<50:raise ValueError('Insufficient nominal data')
 cols,transform=prep(train,list(data.values()));xtrain=transform(train)
 methods=['robust_z','isolation_forest','pca']+(['autoencoder','lstm_autoencoder'] if a.deep else [])
 metrics=[];scores=[]
 for method in methods:
  score=score_deep(method,xtrain,a.epochs) if method in ('autoencoder','lstm_autoencoder') else score_ml(method,xtrain,None)
  threshold=float(np.quantile(score(xtrain),.98))
  for run,df in [('nominal_holdout',hold),('set1',data['set1']),('set2',data['set2'])]:
   s=score(transform(df));pred=(s>threshold).astype(int)
   switches=[f'SWITCH_{i}_STATE' for i in range(8) if f'SWITCH_{i}_STATE' in df]
   truth=np.zeros(len(df),dtype=int) if run=='nominal_holdout' else df[switches].eq(1).any(axis=1).to_numpy(dtype=int)
   pr,re,f1,_=precision_recall_fscore_support(truth,pred,average='binary',zero_division=0)
   tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
   metrics.append(dict(model=method,run=run,tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),precision=pr,recall=re,f1=f1))
   scores.append(pd.DataFrame(dict(model=method,run=run,ccsds_seconds=df.t.to_numpy(),score=s,alert=pred,switch_on_proxy=truth)))
 pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
 pd.concat(scores,ignore_index=True).to_csv(out/'scores.csv',index=False)
 (out/'metadata.json').write_text(json.dumps(dict(features=cols,notes=['Switch-on proxy labels are NOT confirmed cyberattack labels','Independent runs are not matched counterfactuals','ADCS DT differs between nominal and anomaly runs','Scores are not calibrated confidence','No command provenance, authentication, or communication cyber evidence','Point-level evaluation only']),indent=2))
 print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
