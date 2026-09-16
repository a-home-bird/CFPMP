import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pickle
import os
import torch
import pickle
import torch.nn.functional as F
import json



class Interaction(Data,Graph):
    def __init__(self, conf, training, test,dev):
        Graph.__init__(self)
        Data.__init__(self,conf,training,test,dev)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        #self.user_degree = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.dev_set = defaultdict(dict)
        self.dev_set_item = set()


        self.__generate_set()

        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.novel = False
        if conf['model.name'] !='MRLMRP':
            self.ui_adj = self.__create_sparse_bipartite_adjacency_baseline()
            
        else:
            self.ui_adj = self.__create_sparse_bipartite_adjacency(conf)

        if conf.contain('composition_json_path'):
            # Simple composition descriptor experiment
            self.novel = True

            self.mineral_embedding = self.get_composition_descriptor(
                conf['composition_json_path'],
                conf['mineral_id']
            )

            self.m_graph = self.build_topk_normalized_adj(
                self.mineral_embedding,
                int(conf['k'])
            )

            self.R = self.__create_sparse_bipartite_R()
            self.R = self.normalize_graph_mat(self.R)
        elif conf.contain('mineral_embedding_path'):
            self.novel = True
            self.mineral_embedding = self.get_mineral_embedding(conf['mineral_embedding_path'],conf['mineral_dict_path'])
            self.m_graph = self.build_topk_normalized_adj(self.mineral_embedding,int(conf['k']))

            self.R = self.__create_sparse_bipartite_R()
            self.R = self.normalize_graph_mat(self.R)

        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        

    def get_composition_descriptor(self, formula_path, mineral_id_path):
        # mineral name -> element composition
        with open(formula_path, 'r', encoding='utf-8') as f:
            formula_dict = json.load(f)

        # global mineral id -> mineral name
        mineral_dict = self.get_mineral_dict(mineral_id_path)

        # 使用整个 ima_dict 构造固定 element vocabulary
        element_set = set()
        for comp in formula_dict.values():
            element_set.update(comp.keys())

        element_list = sorted(element_set)
        element2id = {e: i for i, e in enumerate(element_list)}

        print("Composition descriptor dimension:", len(element_list))

        result = []
        missing = []

        # 使用 internal item index 顺序，保证与模型 item embedding 完全对应
        for item_idx in range(self.item_num):
            global_id = str(self.id2item[item_idx])

            if global_id not in mineral_dict:
                missing.append(("ID", global_id))
                continue

            mineral_name = mineral_dict[global_id]

            if mineral_name not in formula_dict:
                missing.append(("formula", mineral_name))
                continue

            composition = formula_dict[mineral_name]

            vector = torch.zeros(
                len(element_list),
                dtype=torch.float32
            )

            total = sum(float(v) for v in composition.values())

            if total <= 0:
                raise ValueError(
                    f"Invalid composition for {mineral_name}"
                )

            for element, coefficient in composition.items():
                vector[element2id[element]] = float(coefficient) / total

            result.append(vector)

        if missing:
            print("Missing minerals:")
            for x in missing[:20]:
                print(x)

            raise ValueError(
                f"{len(missing)} minerals cannot be mapped to composition descriptors."
            )

        result = torch.stack(result, dim=0)

        print("Composition matrix:", result.shape)

        return result

    def build_topk_normalized_adj(self,context, k=10, symmetric=False):
        N = context.size(0)

        # 1. 计算相似度
        context_norm = context / torch.norm(context, p=2, dim=-1, keepdim=True)
        sim = torch.mm(context_norm, context_norm.T)  # N×N

        # 2. 去掉对角元素
        sim.fill_diagonal_(-float('inf'))

        # 3. 获取 top-k 索引
        top_val, topk_indices = torch.topk(sim, k=k, dim=-1)  # (N, k)

        # 4. 构造邻接矩阵，值为1
        row_idx = torch.arange(N, device=context.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col_idx = topk_indices.reshape(-1)
        values = torch.ones_like(row_idx, dtype=torch.float32)
        #values = top_val.flatten()

        # 如果对称化则加上反向边
        if symmetric:
            row_idx = torch.cat([row_idx, col_idx], dim=0)
            col_idx = torch.cat([col_idx, row_idx[:len(col_idx)]], dim=0)
            values = torch.ones_like(row_idx, dtype=torch.float32)

        indices = torch.stack([row_idx, col_idx], dim=0)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size([N, N]))

        # 5. 邻接归一化：D^(-1/2) A D^(-1/2)
        deg = torch.sparse.sum(adj, dim=1).to_dense()  # (N,)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        d_row = deg_inv_sqrt[row_idx]
        d_col = deg_inv_sqrt[col_idx]
        norm_values = d_row * values * d_col

        adj_norm = torch.sparse.FloatTensor(indices, norm_values, torch.Size([N, N]))

        return  adj_norm

        

    def get_mineral_dict(self,path=None):

        mineral_dict =  {}
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                line_group = line.rsplit(" ",maxsplit = 1)
                mineral_dict[line_group[1]] = line_group[0]

        return mineral_dict

    def get_mineral_embedding(self,emb_path:str,dict_path:str):
        mineral_cur_dict = self.get_mineral_dict(dict_path)
        mineral_emb = torch.load(emb_path)
        result = []
        try:
            for k,v in self.id2item.items():
                result.append(mineral_emb[mineral_cur_dict[v]])
        except Exception as e:
            print("mineral:{} is not found in mineral_embdding")
        
        if len(result[0].shape) == 2:
            return torch.cat(result,dim=0)
        else:
            return torch.stack(result,dim=0)

    
    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                print("dataset ood error")
                exit()
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)
        
        for entry in self.dev_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                print("dataset ood error")
                exit()
                continue
            self.dev_set[user][item] = rating
            self.dev_set_item.add(item)
    def __create_sparse_bipartite_adjacency_baseline(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # 原始二部图构建
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        

        return adj_mat
    

    def __create_sparse_bipartite_R(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        # 原始二部图构建
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.user_num, self.item_num),dtype=np.float32)
        
        return tmp_adj
    
    
    def __create_sparse_bipartite_adjacency(self, conf, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        """ # 原始二部图构建
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes) """


        #添加虚拟全局节点
        n_nodes = self.user_num + self.item_num

        virtual_node = self.item_num
        vir_node_col = [virtual_node] * self.user_num
        vir_node_raw = list(range(self.user_num))

        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.append(np.array(row_idx),np.array(vir_node_raw))
        item_np = np.append(np.array(col_idx),np.array(vir_node_col))

        

        ratings = np.append(np.ones(len(row_idx), dtype=np.float32),np.ones(len(vir_node_raw), dtype=np.float32))

        #ratings = np.append(ratings,np.ones(len(vir_item_raw), dtype=np.float32) * 0.5)

        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes+1, n_nodes+1),dtype=np.float32)
        #tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes+2, n_nodes+2),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)



        # 添加虚边，效果较差
        """ argument_user = []
        for k,v in self.user_degree.items():
            if v < 3:
                argument_user.append(k)
        item_rating_path = os.path.join(conf['training.set'].rsplit("/", maxsplit=1)[0],'item_rating_matrix.pickle')
        with open(item_rating_path,'rb') as f:
            item_rating = pickle.load(f)
        #item_rating = torch.softmax(item_rating,dim=-1)
        #item_rating = item_rating.ge(0.3).float()
        #index = torch.nonzero(item_rating).numpy()
        val , index = torch.topk(item_rating,k=1)
        index = index.numpy()[argument_user]
        #raw = index[:,0]
        #col = index[:,1]
        raw = np.repeat(np.array(argument_user),1)
        col = index.flatten()
        original = [(self.user[instance[0]],self.item[instance[1]]) for instance in self.training_data]
        argument = [(raw[index],col[index]) for index in range(len(col))]
        argument = list(set(argument) - set(original))
        raw = np.array([i[0] for i in argument])
        col = np.array([i[1] for i in argument])
        val = np.ones(len(col),dtype=np.float32) * 0.1

        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.append(np.array(row_idx),raw)
        item_np = np.append(np.array(col_idx),col)
        ratings = np.append(np.ones(len(row_idx), dtype=np.float32),val)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes) """

        return adj_mat
    
    def update_adj(self,user_embed,item_embed):
        """ argument_user = []
        for k,v in self.user_degree.items():
            if v < 3:
                argument_user.append(k) """
        
        item_rating = torch.matmul(user_embed,item_embed.transpose(0, 1))
        item_rating = torch.softmax(item_rating,dim=-1)
        item_rating = item_rating.ge(0.9).float()
        index = torch.nonzero(item_rating).numpy()
        raw = index[:,0]
        col = index[:,1]

        #val , index = torch.topk(item_rating,k=1)
        #index = index.numpy()[argument_user]
        #raw = np.repeat(np.array(argument_user),1)
        #col = index.flatten()
        original = [(self.user[instance[0]],self.item[instance[1]]) for instance in self.training_data]
        argument = [(raw[index],col[index]) for index in range(len(col))]
        argument = list(set(argument) - set(original))
        raw = np.array([i[0] for i in argument])
        col = np.array([i[1] for i in argument])
        val = np.ones(len(col),dtype=np.float32) * 0.0
        n_nodes = self.user_num + self.item_num
        tmp_adj = sp.csr_matrix((val, (raw, col + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T

        return self.normalize_graph_mat(adj_mat)
        

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
