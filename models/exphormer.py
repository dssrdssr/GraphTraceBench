from __future__ import annotations
from typing import List
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
REAL_EDGE_TYPES=4; EXPANDER_EDGE_TYPE=4; VIRTUAL_EDGE_TYPE=5; SELF_EDGE_TYPE=6; NUM_TOTAL_EDGE_TYPES=7

def grouped_edge_softmax(scores, dst, num_nodes):
    heads = scores.size(1)
    expanded = dst.unsqueeze(-1).expand(-1, heads)
    max_per = torch.full((num_nodes, heads), float('-inf'), device=scores.device, dtype=scores.dtype)
    max_per.scatter_reduce_(0, expanded, scores, reduce='amax', include_self=True)
    exp = (scores - max_per[dst]).exp()
    denom = torch.zeros((num_nodes, heads), device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, dst, exp)
    return exp / denom[dst].clamp_min(1e-12)

def build_expander_local_edges(num_nodes, degree, generator, device):
    if num_nodes <= 1 or degree <= 0:
        return torch.empty((2,0), dtype=torch.long, device=device)
    edges=[]
    for _ in range(degree):
        perm=torch.randperm(num_nodes, generator=generator, device=device); nxt=torch.roll(perm, shifts=-1)
        edges += [torch.stack([perm,nxt],0), torch.stack([nxt,perm],0)]
    return torch.cat(edges,1)

class SparseExphormerAttention(nn.Module):
    def __init__(self, hidden_channels, heads=4, dropout=0.1):
        super().__init__(); assert hidden_channels % heads == 0
        self.hidden_channels=hidden_channels; self.heads=heads; self.head_dim=hidden_channels//heads; self.scale=self.head_dim**-0.5
        self.q=nn.Linear(hidden_channels, hidden_channels); self.k=nn.Linear(hidden_channels, hidden_channels); self.v=nn.Linear(hidden_channels, hidden_channels)
        self.ek=nn.Embedding(NUM_TOTAL_EDGE_TYPES, hidden_channels); self.ev=nn.Embedding(NUM_TOTAL_EDGE_TYPES, hidden_channels)
        self.o=nn.Linear(hidden_channels, hidden_channels); self.drop=nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_type):
        n=x.size(0); src,dst=edge_index
        q=self.q(x).view(n,self.heads,self.head_dim); k=self.k(x).view(n,self.heads,self.head_dim); v=self.v(x).view(n,self.heads,self.head_dim)
        ek=self.ek(edge_type).view(edge_type.size(0),self.heads,self.head_dim); ev=self.ev(edge_type).view(edge_type.size(0),self.heads,self.head_dim)
        mk=k[src]+ek; mv=v[src]+ev; scores=(q[dst]*mk).sum(-1)*self.scale; attn=self.drop(grouped_edge_softmax(scores,dst,n))
        out=torch.zeros((n,self.heads,self.head_dim), device=x.device, dtype=x.dtype); out.index_add_(0,dst,attn.unsqueeze(-1)*mv)
        return self.o(out.reshape(n,self.hidden_channels))
class ExphormerBlock(nn.Module):
    def __init__(self, hidden_channels, heads=4, dropout=0.1):
        super().__init__(); self.n1=nn.LayerNorm(hidden_channels); self.n2=nn.LayerNorm(hidden_channels); self.a=SparseExphormerAttention(hidden_channels,heads,dropout); self.d=nn.Dropout(dropout); self.f=nn.Sequential(nn.Linear(hidden_channels,hidden_channels*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_channels*4,hidden_channels))
    def forward(self, x, edge_index, edge_type):
        x=x+self.d(self.a(self.n1(x), edge_index, edge_type)); x=x+self.d(self.f(self.n2(x))); return x
class ExphormerRegressor(nn.Module):
    def __init__(self, in_channels, edge_dim=4, hidden_channels=128, num_layers=6, heads=4, dropout=0.1, out_channels=1, expander_degree=4, num_virtual_nodes=1, add_real_edges=True, base_expander_seed=12345):
        super().__init__(); self.hidden_channels=hidden_channels; self.num_virtual_nodes=num_virtual_nodes; self.expander_degree=expander_degree; self.add_real_edges=add_real_edges; self.base_expander_seed=base_expander_seed
        self.node_encoder=nn.Sequential(nn.Linear(in_channels,hidden_channels),nn.ReLU(),nn.Linear(hidden_channels,hidden_channels))
        self.virtual_tokens=nn.Embedding(num_virtual_nodes, hidden_channels) if num_virtual_nodes>0 else None
        self.layers=nn.ModuleList([ExphormerBlock(hidden_channels,heads,dropout) for _ in range(num_layers)])
        self.readout=nn.Sequential(nn.Linear(hidden_channels*3,hidden_channels),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_channels,hidden_channels//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_channels//2,out_channels))
    def _augment_graph(self, data, x_real):
        device=x_real.device; batch=data.batch; num_graphs=int(data.num_graphs); num_real=x_real.size(0); x_parts=[x_real]; edge_parts=[]; edge_type_parts=[]
        if self.add_real_edges: edge_parts.append(data.edge_index); edge_type_parts.append(data.edge_type.long())
        if self.num_virtual_nodes>0:
            virtual=self.virtual_tokens.weight.unsqueeze(0).expand(num_graphs,-1,-1).reshape(-1,self.hidden_channels); x_parts.append(virtual)
            for g in range(num_graphs):
                real_nodes=torch.nonzero(batch==g, as_tuple=False).view(-1); start=num_real+g*self.num_virtual_nodes; virt=torch.arange(start,start+self.num_virtual_nodes,device=device)
                s=virt.repeat_interleave(real_nodes.numel()); d=real_nodes.repeat(self.num_virtual_nodes)
                edge_parts += [torch.stack([s,d],0), torch.stack([d,s],0)]
                edge_type_parts += [torch.full((s.numel(),),VIRTUAL_EDGE_TYPE,device=device,dtype=torch.long), torch.full((s.numel(),),VIRTUAL_EDGE_TYPE,device=device,dtype=torch.long)]
        for g in range(num_graphs):
            real_nodes=torch.nonzero(batch==g, as_tuple=False).view(-1); n=real_nodes.numel()
            if n<=1 or self.expander_degree<=0: continue
            gen=torch.Generator(device=device); gen.manual_seed(self.base_expander_seed+int(data.graph_id.view(-1)[g].item()))
            ge=real_nodes[build_expander_local_edges(n,self.expander_degree,gen,device)]
            edge_parts.append(ge); edge_type_parts.append(torch.full((ge.size(1),),EXPANDER_EDGE_TYPE,device=device,dtype=torch.long))
        x_aug=torch.cat(x_parts,0); total=x_aug.size(0); idx=torch.arange(total,device=device)
        edge_parts.append(torch.stack([idx,idx],0)); edge_type_parts.append(torch.full((total,),SELF_EDGE_TYPE,device=device,dtype=torch.long))
        edge_index=torch.cat(edge_parts,1); edge_type=torch.cat(edge_type_parts,0); real_mask=torch.zeros(total,dtype=torch.bool,device=device); real_mask[:num_real]=True
        return x_aug, edge_index, edge_type, real_mask
    def forward(self,data):
        x=self.node_encoder(data.x); x_aug,edge_index,edge_type,real_mask=self._augment_graph(data,x)
        for layer in self.layers: x_aug=layer(x_aug,edge_index,edge_type)
        x=x_aug[real_mask]; pooled=global_mean_pool(x,data.batch); src=data.ptr[:-1]+data.src_id.view(-1); dst=data.ptr[:-1]+data.dst_id.view(-1)
        return self.readout(torch.cat([pooled,x[src],x[dst]],-1))
