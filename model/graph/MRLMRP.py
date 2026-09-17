import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import faiss
import pickle
import os
import numpy as np
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from util.c_graph import build_topk_normalized_adj, sparse_topk_by_row, norm_sparse_adj


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MRLMRP(GraphRecommender):
    def __init__(self, conf, training_set, test_set,dev_set):
        super(MRLMRP, self).__init__(conf, training_set, test_set,dev_set)
        args = OptionConf(self.config['MRLMRP'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_temp = float(args['-tau'])
        self.ssl_reg = float(args['-ssl_reg'])
        self.hyper_layers = int(args['-hyper_layers'])
        self.alpha = float(args['-alpha'])
        self.proto_reg = float(args['-proto_reg'])
        self.k = int(args['-num_clusters'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers,int(conf['k']),float(conf['coe']))
        
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.mse_loss = nn.MSELoss()

    

    
        
    def ssl_layer_loss(self, context_emb, initial_emb, user, item,model):
        
        context_user_emb_all, context_item_emb_all = context_emb[:self.data.user_num], context_emb[self.data.user_num:]
        initial_user_emb_all, initial_item_emb_all = initial_emb[:self.data.user_num], initial_emb[self.data.user_num:]
        
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]

       
        #context_user_emb, initial_user_emb = model.linear_transformer(context_user_emb,initial_user_emb)

        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        

        norm_all_user_emb = F.normalize(initial_user_emb_all)
        
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        
        ssl_loss_user = (-torch.log(pos_score_user)).sum() + torch.logsumexp(ttl_score_user/ self.ssl_temp,dim=-1).sum()

        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]

        #context_item_emb, initial_item_emb = model.linear_transformer(context_item_emb,initial_item_emb)

        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        

        norm_all_item_emb = F.normalize(initial_item_emb_all)
        
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        
        ssl_loss_item = (-torch.log(pos_score_item)).sum() + torch.logsumexp(ttl_score_item/ self.ssl_temp,dim=-1).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        #ssl_loss =  self.ssl_reg * (self.alpha * ssl_loss_item)
        return ssl_loss

    def train(self):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb, emb_list  = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                

                

                ae_loss = self.mse_loss(emb_list[0][user_idx],emb_list[1][user_idx]) + self.mse_loss(emb_list[0][pos_idx],emb_list[1][pos_idx])
                

                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                
                ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,user_idx,pos_idx,model)
                warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)/self.batch_size + ssl_loss + ae_loss * 0.1
                

                if epoch<self.maxEpoch: #warm_up
                    optimizer.zero_grad()
                    warm_up_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
                else:
                    
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _ = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    

class MineralFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, h_v, m_v):
        gate = self.gate(torch.cat([h_v, m_v], dim=-1))
        return gate * h_v + (1 - gate) * m_v

class MineralFusion_2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h_v, m_v):
        gate = self.gate(torch.cat([h_v, m_v], dim=-1))
        return gate * h_v + (1 - gate) * m_v

class MineralFusion_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h_v, m_v):
        feature = self.gate(torch.cat([h_v, m_v], dim=-1))
        return feature
    
class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers,k=0,coe=0):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)
        
        self.novel = False
        if hasattr(self.data, 'm_graph'):
            self.sparse_mineral_adj = self.data.m_graph.to(device)
            self.novel = True
            self.mineral_embs = self.data.mineral_embedding.to(device)
            self.project = nn.Linear(emb_size,emb_size)
            
            self.project_2 = nn.Linear(emb_size,emb_size)
            self.att = nn.Linear(emb_size *2,1)
            self.mf = MineralFusion_3(emb_size)
            self.k = k
            self.coe = coe

            self.embedding_dim = emb_size

            self.gating_weightib=nn.Parameter( 
                torch.FloatTensor(1,self.embedding_dim))
            nn.init.xavier_normal_(self.gating_weightib.data)
            self.gating_weighti=nn.Parameter(
                torch.FloatTensor(self.embedding_dim,self.embedding_dim))
            nn.init.xavier_normal_(self.gating_weighti.data)


            self.mlp_layers = nn.ModuleList()
            for i in range(3):
                self.mlp_layers.append(torch.nn.Linear(self.latent_size, self.latent_size))

    
    def self_gatingi(self,em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weighti) + self.gating_weightib))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num + 1, self.latent_size))),
        })

        
        
        
        
        return embedding_dict

    def forward(self):

        
        

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:-1]

        if self.novel:
            ego_mineral_embs = self.self_gatingi(self.embedding_dict['item_emb'][:-1])           
            all_mineral_embs = [ego_mineral_embs] 
            mineral_embs_prj = F.relu(self.project(self.data.mineral_embedding.to(device)))
            prf_adj = build_topk_normalized_adj(mineral_embs_prj,k=self.k)   
            coe = self.coe
            item_adj = coe * prf_adj + (1-coe) * self.sparse_mineral_adj
            for k in range(1):
                ego_mineral_embs = torch.sparse.mm(item_adj, ego_mineral_embs)
                all_mineral_embs += [ego_mineral_embs]
            all_mineral_embs = all_mineral_embs[-1]
            item_all_embeddings = item_all_embeddings  +   F.normalize(all_mineral_embs,p=2,dim=-1)
            


            
        return user_all_embeddings, item_all_embeddings, all_embeddings
    
    


