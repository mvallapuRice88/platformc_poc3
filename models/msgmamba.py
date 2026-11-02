"""MSGMamba-inspired baseline, NOT an exact published MSGMamba implementation."""
import torch
from torch import nn
from utils import train_sequence,score_sequence
class MSGMambaInspired(nn.Module):
    def __init__(self,features):
        super().__init__();self.graph=nn.Parameter(torch.eye(features));self.proj=nn.Linear(features,48)
        self.convs=nn.ModuleList([nn.Conv1d(48,48,k,padding=k//2) for k in (3,5,7)])
        self.gru=nn.GRU(144,48,batch_first=True);self.out=nn.Linear(48,features)
    def forward(self,x):
        x=torch.relu(self.proj(x@torch.softmax(self.graph,dim=-1))).transpose(1,2)
        x=torch.cat([torch.relu(c(x)).transpose(1,2) for c in self.convs],dim=-1)
        y,_=self.gru(x);return self.out(y[:,-1])

def train(segments,cal,features,w,epochs,fpr,device):return train_sequence(MSGMambaInspired(features),segments,cal,w,epochs,fpr,device)

def predict(model,threshold,x,w,device):
    import numpy as np
    scores=score_sequence(model,x,w,device);valid=np.isfinite(scores)
    return valid,(scores[valid]>threshold).astype(int)
