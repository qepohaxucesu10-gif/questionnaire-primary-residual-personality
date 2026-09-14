"""NOT STUDY DATA. NOT MANUSCRIPT RESULTS."""
from pathlib import Path
import sys,numpy as np,torch
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parents[2]/"src"))
from core import select_items,fuse,SCALES,teacher_top3
from models import QuestionnaireTower,TinyVisualResidual,trainable_parameters
rng=np.random.default_rng(3);x=rng.integers(0,3,(20,187)).astype(float);y=rng.integers(1,11,(20,16)).astype(float);assert x.shape[1]==187;selected=select_items(x,y);q=QuestionnaireTower(len(selected));opt=torch.optim.Adam(q.parameters(),lr=.001,weight_decay=.0001);xb=torch.tensor(x[:2,selected]).float();yb=torch.tensor((y[:2]-1)/9).float();loss=torch.nn.functional.mse_loss(q(xb),yb);opt.zero_grad();loss.backward();opt.step();assert q(xb).shape==(2,16);tiny=TinyVisualResidual();out=tiny(torch.rand(2,1,128,128));assert out.shape==(2,16) and torch.all(out<=1) and torch.all(out>=-1);assert trainable_parameters(tiny)==19648;residual=y[:2]-(1+9*q(xb).detach().numpy());assert fuse(y[:2],out.detach().numpy()*9,.05).shape==(2,16);assert list(SCALES)==[0,.005,.01,.025,.05,.075,.1];assert len(teacher_top3(y[0]))==3;print("Synthetic smoke test passed. NOT STUDY DATA. NOT MANUSCRIPT RESULTS.")
