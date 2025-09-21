import torch
import torch.nn.functional as F
from torch import nn

Actor_in_features=10          # state_dim * 2 = 5 * 2 = 10
Actor_hidden_features1=32
Actor_hidden_features2=16

Critic_in_features=20         # state_dim * 2 + action_dim * 2 = 10 + 10 = 20 (GAT-MF attention applied to both)  
Critic_hidden_features1=32
Critic_hidden_features2=16

Attention_in_features=5       # state_dim = 5
Attention_hidden_features=32

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()

        self.lin1=nn.Linear(in_features=Actor_in_features,out_features=Actor_hidden_features1)
        self.lin2=nn.Linear(in_features=Actor_hidden_features1,out_features=Actor_hidden_features2)
        self.lin3=nn.Linear(in_features=Actor_hidden_features2,out_features=5)

    def forward(self,s):
        s=F.leaky_relu(self.lin1(s))
        s=F.leaky_relu(self.lin2(s))
        s=self.lin3(s)

        return s

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()

        self.lin1=nn.Linear(in_features=Critic_in_features,out_features=Critic_hidden_features1)
        self.lin2=nn.Linear(in_features=Critic_hidden_features1,out_features=Critic_hidden_features2)
        self.lin3=nn.Linear(in_features=Critic_hidden_features2,out_features=1)

    def forward(self,s,a):
        x=torch.concat((s,a),dim=2)
        x=F.leaky_relu(self.lin1(x))
        x=F.leaky_relu(self.lin2(x))
        x=self.lin3(x)

        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()

        self.Qweight=nn.Parameter(torch.rand(Attention_in_features,Attention_hidden_features)*((4/Attention_in_features)**0.5)-(1/Attention_in_features)**0.5)
        self.Kweight=nn.Parameter(torch.rand(Attention_in_features,Attention_hidden_features)*((4/Attention_in_features)**0.5)-(1/Attention_in_features)**0.5)

    def forward(self,s,Gmat):
        q=torch.einsum('ijk,km->ijm',s,self.Qweight)
        k=torch.einsum('ijk,km->ijm',s,self.Kweight).permute(0, 2, 1)

        att=torch.square(torch.bmm(q,k))*Gmat
        att=att/(torch.sum(att,dim=2,keepdim=True)+0.001)

        return att