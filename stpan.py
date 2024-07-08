import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
import os

class Embedding(nn.Module):
    def __init__(self, node, hidden, input_dim, input_len, output_len):
        super(Embedding, self).__init__()
        
        self.node_emb = nn.Parameter(torch.empty(node, hidden))
        nn.init.xavier_uniform_(self.node_emb)

        self.time_emb = nn.Parameter(torch.empty(288, hidden))
        nn.init.xavier_uniform_(self.time_emb)

        self.day_emb = nn.Parameter(torch.empty(7, hidden))
        nn.init.xavier_uniform_(self.day_emb)

        self.pos_emb = nn.Parameter(torch.empty(input_len, hidden))
        nn.init.xavier_uniform_(self.pos_emb)
        
        self.dense = nn.ModuleList([nn.Linear(input_len, hidden) for _ in range(input_len)])
        self.conv1 = nn.Conv2d(in_channels=input_len, out_channels=hidden, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden*4, out_channels=hidden, kernel_size=(1, 1), bias=True)
        self.linear1 = nn.Linear(hidden*5, hidden)
    def forward(self, x):
        batch, in_step, node, _ = x.size()
        #x: batch,step,node,,hidden

        spatial_emb = self.node_emb.unsqueeze(0).expand(batch, -1, -1)       
        #spatial_emb: batch,node,hidden

        input_data = x[..., 0].unsqueeze(-1)
        #input_data: batch, step，node

        input_emb_p = self.conv1(input_data).squeeze(-1).transpose(1, 2)
        #input_emb_p: batch,node，hidden      
        
        input_emb_r = []
        input_r = x[..., 0].permute(0,2,1)
        for net in self.dense:
            input_emb_r.append(net(input_r).unsqueeze(1))
        input_emb_r = torch.cat(input_emb_r,dim=1)
        
        #input_emb_r: batch,step,node,hidden

        day_id = x[..., 2]
        temporal_day_emb = self.day_emb[(day_id[:,-1,:]).type(torch.LongTensor)]
        #temporal_day_emb: batch,node,hidden
       
        
        time_id = x[..., 1]
        temporal_time_emb = self.time_emb[(time_id[:,-1,:]).type(torch.LongTensor)]
        #temporal_time_emb: batch,node,hidden

        position_emb = self.pos_emb.unsqueeze(0).unsqueeze(2).expand(batch, -1, node, -1)
        #position_emb: batch,step,node,hidden

        p = torch.cat([temporal_time_emb, temporal_day_emb, input_emb_p, spatial_emb], dim=-1)
        p = p.transpose(1,2).unsqueeze(-1)
        p = self.conv2(p)
        #p: batch,hidden,node,1
        r = torch.cat([temporal_time_emb.unsqueeze(1).expand(-1,in_step,-1,-1), temporal_day_emb.unsqueeze(1).expand(-1,in_step,-1,-1), 
                       input_emb_r, spatial_emb.unsqueeze(1).expand(-1,in_step,-1,-1), position_emb], dim=-1)
        r = self.linear1(r)
        #r: batch,step,node,hidden
        return p, r

class Res_Memory(nn.Module):
    def __init__(self, m, input_dim, hidden_dim, device):
        super(Res_Memory, self).__init__()
        self.m1 = nn.Parameter(torch.empty(m, hidden_dim))
        nn.init.xavier_uniform_(self.m1)
        self.m_size = m
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.relu = nn.ReLU()
    def forward(self, input):
        # input: batch, hidden, node, 1

        emb = self.relu(self.conv1(input)).squeeze()
        # batch, hidden, node     #

        m = self.m1.softmax(-1)
        m1_index = torch.sum(m * torch.log(m) - m * torch.log(emb.permute(0,2,1).softmax(-1).unsqueeze(2)), dim=-1).argmin(dim=-1)
        # m1_index = torch.sum(self.m1 * (torch.log(self.m1)-torch.log(emb.permute(0,2,1).unsqueeze(2))), dim=-1).argmin(dim=-1)



        # m1_index = torch.sum(self.m1 * (torch.log(self.m1)-torch.log(emb.permute(0,2,1).unsqueeze(2))), dim=-1).argmin(dim=-1)
        #m1_index: batch, node, 1
        
        # if self.training is True:
        #     self.m1[m1_index.type(torch.LongTensor)] = 0.999*self.m1[m1_index.type(torch.LongTensor)] + 0.001*emb.permute(0,2,1)

        m1 = self.m1[m1_index.type(torch.LongTensor)].permute(0,2,1)
        #batch, hidden, node
        m1 = emb - m1
        out = self.conv2(torch.cat([emb,m1],dim=1).unsqueeze(-1))
        # out: batch, hidden, node, 1
        return out

class Graph_Memory_Attn(nn.Module):
    def __init__(self, hidden, head, node, device, m):
        super(Graph_Memory_Attn, self).__init__()

        self.memeory = Memory(m, hidden, hidden, device)
        self.res_mem = Res_Memory(m, hidden, hidden, device)
        self.h = hidden
        self.head = head
        self.node = node
        self.q = nn.Linear(hidden, hidden*head)
        self.k = nn.Linear(hidden, hidden*head)
        self.v = nn.Linear(hidden, hidden*head)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.stru = nn.Parameter(torch.empty(node, node))
        self.eye = torch.eye(node, device=device)
        self.hardsigmoid = nn.Hardsigmoid()

        self.dense1 = nn.Linear(hidden*head, hidden)
        nn.init.xavier_uniform_(self.stru)

    def forward(self, input, topo):
        #x: batch,step,node,hidden
        out = 0
        batch, in_step, node, _ = input.size()
        for i in range(in_step):
            x = input[:,i,:,:].unsqueeze(1).permute(0,3,2,1)
            #x: batch, hidden, node, 1
            
            mem = self.memeory(x).squeeze(-1).permute(0,2,1)
            #mem: batch, node, hidden

            q = self.relu(self.q(mem)).reshape(batch,node,self.h,self.head).permute(0,3,1,2)
            #q: batch,head,node,hidden
            k = self.relu(self.k(mem)).reshape(batch,node,self.h,self.head).permute(0,3,2,1)
            #k: batch,head,hidden,node
            qk = torch.matmul(q, k)/math.sqrt(self.h)
            #qk: batch,head,node,node


            stru = self.hardsigmoid(self.stru)

            qk = qk * stru
            # zero_vec = -9e15*torch.ones_like(qk)
            # qk = torch.where(topo > 0, qk, zero_vec)
            attn_score = torch.softmax(qk,dim=-1)
            #attn_score: batch,head,node,node

            
            res_mem = self.res_mem(x).squeeze(-1).permute(0,2,1)
            #res_mem: batch,node,hidden

            v = self.relu(self.v(res_mem)).reshape(batch,node,self.h,self.head).permute(0,3,1,2)
            #v: batch,head,node,hidden

            attn = torch.matmul(attn_score, v)
            #attn: batch,node,hidden
            out += attn
        
        out = out.permute(0,2,1,3).reshape(-1,self.node,self.h*self.head)
        out = self.drop(self.relu(self.dense1(out)))
        out = out.permute(0,2,1).unsqueeze(-1)
        #out: batch,hidden,node,1
        return out





class Residual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        # batch,         
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
    def forward(self, input_data) -> torch.Tensor:
        hidden = self.relu(self.conv1(input_data))
        hidden = self.drop(hidden)
        hidden = self.conv2(hidden)         
        out = hidden + input_data
        return out


class Memory(nn.Module):
    def __init__(self, m, input_dim, hidden_dim, device):
        super(Memory, self).__init__()
        self.m1 = nn.Parameter(torch.empty(m, hidden_dim))
        self.m_size = m
        nn.init.xavier_uniform_(self.m1)
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.relu = nn.ReLU()
    def forward(self, input):
        # input: batch, hidden, node, 1

        emb = self.relu(self.conv1(input)).squeeze()
        # batch, hidden, node     #
        m = self.m1.softmax(-1)
        m1_index = torch.sum(m * torch.log(m) - m * torch.log(emb.permute(0,2,1).softmax(-1).unsqueeze(2)), dim=-1).argmin(dim=-1)

        # m1_index = torch.sum(self.m1 * (torch.log(self.m1)-torch.log(emb.permute(0,2,1).unsqueeze(2))), dim=-1).argmin(dim=-1)
        # m1_index = torch.sum(self.m1 * torch.log(self.m1) - self.m1 * torch.log(emb.permute(0,2,1).unsqueeze(2)), dim=-1).argmin(dim=-1)
        # m1_index = torch.sum(self.m1 * (torch.log(self.m1)-torch.log(emb.permute(0,2,1).unsqueeze(2))), dim=-1).argmin(dim=-1)
        #m1_index: batch, node, 1

        m1 = self.m1[m1_index.type(torch.LongTensor)].permute(0,2,1)
        #batch, hidden, node

        out = self.conv2(torch.cat([emb,m1],dim=1).unsqueeze(-1))
        # out: batch, hidden, node, 1
        return out

class Period(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Period, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)     
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data):
        hidden = self.relu(self.conv1(input_data))
        hidden = self.drop(hidden)
        hidden = self.conv2(hidden)      
        out = hidden + input_data
        return out

class STPAN(nn.Module):

    def __init__(self,node,hidden,head,num_Period_Block,input_length,output_length,m,device,topo):
        super(STPAN, self).__init__()
        self.topo = topo
        self.embedding1 = Embedding(node, hidden, 3, input_length, output_length)

        self.memory = Memory(m, hidden, hidden, device)

        self.gma = Graph_Memory_Attn(hidden, head, node, device, m)

        self.residual = nn.ModuleList([Residual(hidden, hidden) for _ in range(num_Period_Block)])
        
        self.period = nn.ModuleList([Period(hidden, hidden) for _ in range(num_Period_Block)])


        self.prediction = nn.Conv2d(in_channels=hidden, out_channels=output_length, kernel_size=(1, 1), bias=True)


    def forward(self, x):
        #x: batch,step,node,3
        p, r = self.embedding1(x)

        p = self.memory(p)
        #p: batch, hidden, node, 1
        for net in self.period:
            p = net(p)
        # p = self.memory(p)
        #P: batch, hidden, node, 1
        
        #r: batch, step, node, hidden
        r = self.gma(r, self.topo)
        #attn: batch,hidden,node,1
        for net in self.residual:
            r = net(r)


        o = r + p
        #o: batch, hidden, node, 1
        
        o = self.prediction(o).squeeze(-1)
        return o
    
