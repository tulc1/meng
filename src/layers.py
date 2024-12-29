import warnings
import scipy.sparse as sp

warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp
import numpy as np
from scipy.signal import convolve2d
import time
import collections
from abc import abstractmethod

# tf.enable_eager_execution()

import argparse
from data_loader import load_data

LAYER_IDS = {}



def get_layer_id(layer_name=''):
    #print('开始get_layer_id')
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Layer(object):
    def __init__(self, name):
        #print('开始Layer的__init__')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        #print(self.name)
        self.vars = []

    def __call__(self, inputs):
        #print('开始Layer的__call__')
        outputs = self._call(inputs)
        return outputs

    @abstractmethod
    def _call(self, inputs):
        #print('开始Layer的__call')
        pass


# 使用tf.layers.Dense方法定义了不带偏置的全连接层，将该全连接层作用于交叉后的特征向量，实现压缩的过程
class Dense(Layer):
    def __init__(self, args, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, name=None):
        #print('定义不带偏置的全连接层，将该全连接层作用于交叉后的特征向量，实现压缩的过程')
        super(Dense, self).__init__(name)

        data = load_data(args)
        self.n_user, self.n_item, self.n_entity, n_relation = data[0], data[1], data[2], data[3]
        self.train_data, eval_data, test_data = data[4], data[5], data[6]
        self.p = args.p

        file_name = '../data/' + args.dataset + '/kg_final.txt'
        kg_np = np.loadtxt(file_name, dtype=np.int32)
        self.kg_np = np.unique(kg_np, axis=0)
        #print('kg_np文件读取成功')
        n_relations = max(self.kg_np[:, 1]) + 1

        kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)

        for head, relation, tail in self.kg_np:
            kg_dict[head].append((tail, relation))
            self.relation_dict[relation].append((head, tail))


        self.n_fold=100
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.dataset=args.dataset
        self.all_weights = dict()
        initializer = tf.keras.initializers.glorot_normal(seed=1)
        if self.input_dim==64:
            emb_dim,kge_dim=32,32
            self.weight_size = eval('[16,8,8]')
        elif self.input_dim==128:
            emb_dim,kge_dim=64,64
            self.weight_size = eval('[32,16,16]')
        elif self.input_dim==256:
            emb_dim,kge_dim=128,128
            self.weight_size = eval('[64,32,32]')
        elif self.input_dim==512:
            emb_dim,kge_dim=256,256
            self.weight_size = eval('[128,64,64]')
        elif self.input_dim==1024:
            emb_dim,kge_dim=512,512
            self.weight_size = eval('[256,128,128]')
        elif self.input_dim==32:
            emb_dim,kge_dim=16,16
            self.weight_size = eval('[8,4,4]')
        elif self.input_dim==16:
            emb_dim,kge_dim=8,8
            self.weight_size = eval('[4,2,2]')
        elif self.input_dim==2048:
            emb_dim,kge_dim=1024,1024
            self.weight_size = eval('[512,256,256]')

        
        self.n_layers = len(self.weight_size)
        self.all_weights['item_embed'] = tf.Variable(initializer([self.n_item, emb_dim]), name='item_embed')# shape=(6036,8)        
        self.all_weights['user_embed'] = tf.Variable(initializer([self.n_user, emb_dim]), name='user_embed')# shape=(6036,8)
        self.all_weights['entity_embed'] = tf.Variable(initializer([self.n_entity, emb_dim]), name='entity_embed')# shape=(6729,8)

        self.all_weights['relation_embed'] = tf.Variable(initializer([n_relations, kge_dim]),
                                                    name='relation_embed')# shape=(7,8)
        self.all_weights['trans_W'] = tf.Variable(initializer([n_relations, emb_dim, kge_dim]))# shape=(7,8,8)

        self.weight_size_list = [emb_dim] + self.weight_size
        for k in range(self.n_layers):
            self.all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            self.all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            self.all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            self.all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

        with tf.variable_scope(self.name):
            self.weight_vv = tf.get_variable(name='weight_vv', shape=(self.input_dim, 1), dtype=tf.float32)
            self.weight_ev = tf.get_variable(name='weight_ev', shape=(self.input_dim, 1), dtype=tf.float32)
            self.weight_ve = tf.get_variable(name='weight_ve', shape=(self.input_dim, 1), dtype=tf.float32)
            self.weight_ee = tf.get_variable(name='weight_ee', shape=(self.input_dim, 1), dtype=tf.float32)
            self.bias_v = tf.get_variable(name='bias_v', shape=self.input_dim, initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e', shape=self.input_dim, initializer=tf.zeros_initializer())

        # self.vars = [self.weight]
        self.vars = [self.all_weights['W_gc_0'],self.all_weights['b_gc_0'],self.all_weights['W_bi_0'],self.all_weights['b_bi_0'],
        self.all_weights['W_gc_1'],self.all_weights['b_gc_1'],self.all_weights['W_bi_1'],self.all_weights['b_bi_1'],
        self.all_weights['W_gc_2'],self.all_weights['b_gc_2'],self.all_weights['W_bi_2'],self.all_weights['b_bi_2'],
        self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee]


    
    def _split_A_hat(self, X):
        #print('开始KGAT中的_split_A_hat')
        A_fold_hat = []

        fold_len = (self.n_user + self.n_entity) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_user + self.n_entity
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _convert_sp_mat_to_sp_tensor(self, X):
        # print('开始KGAT中的_convert_sp_mat_to_sp_tensor')
        coo = X.tocoo().astype(np.float32)
        # mat创建数组
        indices = np.mat([coo.row, coo.col]).transpose()
        # tf.SparseTensor生成稀疏矩阵
        return tf.SparseTensor(indices, coo.data, coo.shape)


    def _si_norm_lap(self,adj):
            # print('开始loader_kgat中的_si_norm_lap')
        
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            #将所有inf转为0
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

    def _get_relational_adj_list(self):
        #print('为用户项目交互和关系KG数据生成稀疏邻接矩阵')
        
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            # print('生成矩阵，然后返回的矩阵互为转置矩阵')
            # 6036+6729
            n_all = self.n_user + self.n_entity
        
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre

            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        #print('生成CF的稀疏矩阵：R,R_inv')
        # user-item的邻接矩阵 R和R_inv互逆
        train_data1 = list()
        for interaction in self.train_data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if label == 1:
                train_data1.append([user,item])
        R, R_inv = _np_mat2sp_adj(np.array(train_data1), row_pre=0, col_pre=self.n_user)

        adj_mat_list.append(R)

        adj_r_list.append(0)

        adj_mat_list.append(R_inv)

        n_relations = max(self.kg_np[:, 1]) + 1
        adj_r_list.append(n_relations - 1)
        #print('\tconvert ratings into adj mat done.')

        #print('生成KG的稀疏矩阵：K,K_inv')
        for r_id in self.relation_dict.keys():
            #np.array(self.relation_dict[r_id])==>[（head,tail）,(head,tail),...]
            #K:[[129955*129955矩阵(但是rows∈[23566,129954] cols∈[23566,129953]才有值）]]
            #item-item邻接矩阵
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_user, col_pre=self.n_user)
            #adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list.append(K)
            #adj_r_list[??] = [1,2,3,4,5,6,7,8,9]
            adj_r_list.append(r_id)

            # adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list.append(K_inv)
            # adj_r_list[??] = [11,12,13,14,15,16,17,18,19]
            adj_r_list.append(r_id)
        #20 有向图，所以关系也加了一倍
        #print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list


    def _get_relational_adj_list1(self):
        #print('为用户项目交互和关系KG数据生成稀疏邻接矩阵')
        
        adj_mat_list1 = []
        adj_r_list1 = []

        def _np_mat2sp_adj1(np_mat, row_pre, col_pre):
            # print('生成矩阵，然后返回的矩阵互为转置矩阵')
            # 6036+6729
            n_all1 = self.n_user + self.n_item
        
            a_rows1 = np_mat[:, 0]
            a_cols1 = np_mat[:, 1]

            a_vals1 = [1.] * len(a_rows1)

            b_rows1 = a_cols1
            b_cols1 = a_rows1
            b_vals1 = [1.] * len(b_rows1)

            a_adj = sp.coo_matrix((a_vals1, (a_rows1, a_cols1)), shape=(n_all1, n_all1))
            b_adj = sp.coo_matrix((b_vals1, (b_rows1, b_cols1)), shape=(n_all1, n_all1))

            return a_adj, b_adj

        #print('生成CF的稀疏矩阵：R,R_inv')
        # user-item的邻接矩阵 R和R_inv互逆
        train_data1 = list()
        for interaction in self.train_data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if label == 1:
                train_data1.append([user,item])
        R, R_inv = _np_mat2sp_adj1(np.array(train_data1), row_pre=0, col_pre=self.n_user)

        adj_mat_list1.append(R)

        adj_r_list1.append(0)

        adj_mat_list1.append(R_inv)

        n_relations = max(self.kg_np[:, 1]) + 1
        adj_r_list1.append(n_relations - 1)
        #print('\tconvert ratings into adj mat done.')

        #print('生成KG的稀疏矩阵：K,K_inv')
        for r_id in self.relation_dict.keys():
            #np.array(self.relation_dict[r_id])==>[（head,tail）,(head,tail),...]
            #K:[[129955*129955矩阵(但是rows∈[23566,129954] cols∈[23566,129953]才有值）]]
            #item-item邻接矩阵
            K, K_inv = _np_mat2sp_adj1(np.array(self.relation_dict[r_id]), row_pre=self.n_user, col_pre=self.n_user)
            #adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list1.append(K)
            #adj_r_list[??] = [1,2,3,4,5,6,7,8,9]
            adj_r_list1.append(r_id)

            # adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list1.append(K_inv)
            # adj_r_list[??] = [11,12,13,14,15,16,17,18,19]
            adj_r_list1.append(r_id)
        #20 有向图，所以关系也加了一倍
        #print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        n_relations = len(adj_r_list1)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list1, adj_r_list1



    def _get_relational_adj_list2(self):
        #print('为用户项目交互和关系KG数据生成稀疏邻接矩阵')
        
        adj_mat_list2 = []
        adj_r_list2 = []

        def _np_mat2sp_adj2(np_mat, row_pre, col_pre):
            # print('生成矩阵，然后返回的矩阵互为转置矩阵')
            # 6036+6729
            n_all2 = self.n_entity + self.n_item
        
            a_rows2 = np_mat[:, 0]
            a_cols2 = np_mat[:, 1]

            a_vals2 = [1.] * len(a_rows2)

            b_rows2 = a_cols2
            b_cols2 = a_rows2
            b_vals2 = [1.] * len(b_rows2)

            a_adj = sp.coo_matrix((a_vals2, (a_rows2, a_cols2)), shape=(n_all2, n_all2))
            b_adj = sp.coo_matrix((b_vals2, (b_rows2, b_cols2)), shape=(n_all2, n_all2))

            return a_adj, b_adj

        #print('生成CF的稀疏矩阵：R,R_inv')
        # user-item的邻接矩阵 R和R_inv互逆
        train_data1 = list()
        for interaction in self.train_data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if label == 1:
                train_data1.append([user,item])
        R, R_inv = _np_mat2sp_adj2(np.array(train_data1), row_pre=0, col_pre=self.n_item)

        adj_mat_list2.append(R)

        adj_r_list2.append(0)

        adj_mat_list2.append(R_inv)

        n_relations = max(self.kg_np[:, 1]) + 1
        adj_r_list2.append(n_relations - 1)
        #print('\tconvert ratings into adj mat done.')

        #print('生成KG的稀疏矩阵：K,K_inv')
        for r_id in self.relation_dict.keys():
            #np.array(self.relation_dict[r_id])==>[（head,tail）,(head,tail),...]
            #K:[[129955*129955矩阵(但是rows∈[23566,129954] cols∈[23566,129953]才有值）]]
            #item-item邻接矩阵
            K, K_inv = _np_mat2sp_adj2(np.array(self.relation_dict[r_id]), row_pre=self.n_item, col_pre=self.n_item)
            #adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list2.append(K)
            #adj_r_list[??] = [1,2,3,4,5,6,7,8,9]
            adj_r_list2.append(r_id)

            # adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list2.append(K_inv)
            # adj_r_list[??] = [11,12,13,14,15,16,17,18,19]
            adj_r_list2.append(r_id)
        #20 有向图，所以关系也加了一倍
        #print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        n_relations = len(adj_r_list2)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list2, adj_r_list2

    def _get_relational_adj_list3(self):
        #print('为用户项目交互和关系KG数据生成稀疏邻接矩阵')
        
        adj_mat_list3 = []
        adj_r_list3 = []

        def _np_mat2sp_adj3(np_mat, row_pre, col_pre):
            # print('生成矩阵，然后返回的矩阵互为转置矩阵')
            # 6036+6729
            n_all3 = self.n_user + self.n_user
        
            a_rows3 = np_mat[:, 0]
            a_cols3 = np_mat[:, 1]

            a_vals3 = [1.] * len(a_rows3)

            b_rows3 = a_cols3
            b_cols3 = a_rows3
            b_vals3 = [1.] * len(b_rows3)

            a_adj = sp.coo_matrix((a_vals3, (a_rows3, a_cols3)), shape=(n_all3, n_all3))
            b_adj = sp.coo_matrix((b_vals3, (b_rows3, b_cols3)), shape=(n_all3, n_all3))

            return a_adj, b_adj

        #print('生成CF的稀疏矩阵：R,R_inv')
        # user-item的邻接矩阵 R和R_inv互逆
        train_data1 = list()
        for interaction in self.train_data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if label == 1:
                train_data1.append([user,item])
        R, R_inv = _np_mat2sp_adj3(np.array(train_data1), row_pre=0, col_pre=self.n_user)

        adj_mat_list3.append(R)

        adj_r_list3.append(0)

        adj_mat_list3.append(R_inv)

        n_relations = max(self.kg_np[:, 1]) + 1
        adj_r_list3.append(n_relations - 1)
        #print('\tconvert ratings into adj mat done.')

        #print('生成KG的稀疏矩阵：K,K_inv')
        for r_id in self.relation_dict.keys():
            #np.array(self.relation_dict[r_id])==>[（head,tail）,(head,tail),...]
            #K:[[129955*129955矩阵(但是rows∈[23566,129954] cols∈[23566,129953]才有值）]]
            #item-item邻接矩阵
            K, K_inv = _np_mat2sp_adj3(np.array(self.relation_dict[r_id]), row_pre=self.n_user, col_pre=self.n_user)
            #adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list3.append(K)
            #adj_r_list[??] = [1,2,3,4,5,6,7,8,9]
            adj_r_list3.append(r_id)

            # adj_mat_list[??]=[129955*129955矩阵]
            adj_mat_list3.append(K_inv)
            # adj_r_list[??] = [11,12,13,14,15,16,17,18,19]
            adj_r_list3.append(r_id)
        #20 有向图，所以关系也加了一倍
        #print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        n_relations = len(adj_r_list3)
        # print('\t adj relation list is', adj_r_list)

        return adj_mat_list3, adj_r_list3
    

    def _compute_reference_adj(self, node_features, threshold_p):
       
        normed_features = tf.math.l2_normalize(node_features, axis=1)  
        similarity_matrix = tf.matmul(normed_features, normed_features, transpose_b=True)  

        reference_adj = tf.where(similarity_matrix > threshold_p, tf.ones_like(similarity_matrix), tf.zeros_like(similarity_matrix))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  
            reference_adj_np = sess.run(reference_adj)  
            
        reference_adj_sparse = sp.coo_matrix(reference_adj_np)
        return reference_adj_sparse
    

    def _augment_graph(self, adj, node_features):
        """
        图结构和节点特征增强
        
        # 边扰动：随机删除或添加边
        coo = adj.tocoo()
        row, col = coo.row, coo.col

        # 随机删除一定比例的边
        mask = np.random.choice([True, False], size=len(row), p=[0.9, 0.1])
        row, col = row[mask], col[mask]

        # 随机增加一定比例的边
        num_additional_edges = int(len(row) * 0.05)
        additional_row = np.random.randint(0, adj.shape[0], num_additional_edges)
        additional_col = np.random.randint(0, adj.shape[1], num_additional_edges)

        # 合并原始和新增的边
        new_row = np.concatenate([row, additional_row])
        new_col = np.concatenate([col, additional_col])
        new_data = np.ones_like(new_row, dtype=np.float32)
        augmented_adj = sp.coo_matrix((new_data, (new_row, new_col)), shape=adj.shape)
        """
        feature_shape = tf.shape(node_features)  

        # 随机扰动节点特征
        noise = tf.random.normal(feature_shape, mean=0.0, stddev=0.1)
        perturbed_features = node_features + noise

        perturbed_features = tf.math.l2_normalize(perturbed_features, axis=1)

        return adj, perturbed_features

    def _gcn(self, adj, embeddings):

        adj_norm = self._normalize_adj(adj)

        print("adj_norm dense_shape: ", adj_norm.dense_shape)
        print("features shape: ", embeddings)

        all_embeddings = [embeddings]
        for _ in range(2):      
            embeddings = tf.sparse.sparse_dense_matmul(adj_norm, embeddings)
            all_embeddings.append(embeddings)
        print("gcn_features shape: ", tf.reduce_mean(tf.stack(all_embeddings, axis=0), axis=0).shape)

        return tf.reduce_mean(tf.stack(all_embeddings, axis=0), axis=0)

    def _normalize_adj(self, adj):

        adj = adj + sp.eye(adj.shape[0])  
        rowsum = np.array(adj.sum(1))  
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

       
        coo = adj_normalized.tocoo()
        adj_tf = tf.sparse.SparseTensor(
            indices=np.vstack((coo.row, coo.col)).T,
            values=coo.data.astype(np.float32), 
            dense_shape=coo.shape
        )
        adj_tf = tf.sparse.reorder(adj_tf)  
        return adj_tf


    def _call(self, inputs):

        self.adj_list, self.adj_r_list = self._get_relational_adj_list() # len(self.adj_list) 16

        lap_list = [self._si_norm_lap(adj) for adj in self.adj_list]  # len(lap_list) 16
        self.A = sum(lap_list)
        self.A_fold_hat = self._split_A_hat(self.A)# 列表，存储了分割之后的稀疏矩阵

        ego_embeddings = tf.concat([self.all_weights['user_embed'], self.all_weights['entity_embed']], axis=0)# shape=(12765,8)
        all_embeddings = [ego_embeddings]# shape=(12765,16)加括号是为了后面的matmul运算
        print("ego_embeddings ", ego_embeddings.shape)

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):# 100
                # 稀疏tensor和稠密矩阵相乘的方法，完成聚合过程
                temp_embed.append(tf.sparse_tensor_dense_matmul(self.A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)# shape=(12765,8)

            add_embeddings = ego_embeddings + side_embeddings# shape=(12765,8)

            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(add_embeddings, self.all_weights['W_gc_%d' % k]) + self.all_weights['b_gc_%d' % k])# shape=(12765,8)


            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)# shape=(12765,8)

            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.all_weights['W_bi_%d' % k]) + self.all_weights['b_bi_%d' % k])# shape=(12765,8)

            ego_embeddings = bi_embeddings + sum_embeddings# shape=(12765,8)

            ego_embeddings = tf.nn.dropout(ego_embeddings, 0.9)# shape=(12765,8)

            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)# shape=(12765,8)

            all_embeddings += [norm_embeddings]# shape=(12765,16)

        all_embeddings = tf.concat(all_embeddings, 1)# shape=(12765,16)
        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_user, self.n_entity], 0)

########1
        if self.dataset!='music':
        # if True:
            self.adj_list1, self.adj_r_list1 = self._get_relational_adj_list1()

            lap_list1 = [self._si_norm_lap(adj) for adj in self.adj_list1]
            self.A1 = sum(lap_list1)

            self.A_fold_hat1 = self._split_A_hat(self.A1)
            ego_embeddings1 = tf.concat([self.all_weights['user_embed'], self.all_weights['item_embed']], axis=0)
            all_embeddings1 = [ego_embeddings1]
            
            for k in range(0, self.n_layers-2):

                temp_embed1 = []
                for f in range(self.n_fold):
                    temp_embed1.append(tf.sparse_tensor_dense_matmul(self.A_fold_hat1[f], ego_embeddings1))

                side_embeddings1 = tf.concat(temp_embed1, 0)
                ego_embeddings1 = side_embeddings1
                all_embeddings1 += [ego_embeddings1]
            all_embeddings1 = tf.concat(all_embeddings1, 1)
            u_g_embeddings, ui_embeddings = tf.split(all_embeddings1, [self.n_user, self.n_item], 0)
            # shape=(nums,dim)

            ua_e = ua_embeddings + u_g_embeddings
            print("结束user_item_gcn")


########3
        if self.dataset!='music':
        # if True:
            self.adj_list3, self.adj_r_list3 = self._get_relational_adj_list3()

            lap_list3 = [self._si_norm_lap(adj) for adj in self.adj_list3]
            self.A3 = sum(lap_list3)

            self.A_fold_hat3 = self._split_A_hat(self.A3)
            ego_embeddings3 = tf.concat([self.all_weights['user_embed'], self.all_weights['user_embed']], axis=0)
            all_embeddings3 = [ego_embeddings3]
            
            for k in range(0, self.n_layers):

                temp_embed3 = []
                for f in range(self.n_fold):
                    temp_embed3.append(tf.sparse_tensor_dense_matmul(self.A_fold_hat3[f], ego_embeddings3))

                side_embeddings3 = tf.concat(temp_embed3, 0)
                add_embeddings3 = ego_embeddings3 + side_embeddings3
                sum_embeddings3 = tf.nn.leaky_relu(tf.matmul(add_embeddings3, self.all_weights['W_gc_%d' % k]) + self.all_weights['b_gc_%d' % k])# shape=(12765,8)
                bi_embeddings3 = tf.multiply(ego_embeddings3, side_embeddings3)
                bi_embeddings3 = tf.nn.leaky_relu(tf.matmul(bi_embeddings3, self.all_weights['W_bi_%d' % k]) + self.all_weights['b_bi_%d' % k])# shape=(12765,8)
                ego_embeddings3 = bi_embeddings3 + sum_embeddings3
                ego_embeddings3 = tf.nn.dropout(ego_embeddings3, 0.9)
                norm_embeddings3 = tf.math.l2_normalize(ego_embeddings3, axis=1)
            
                all_embeddings3 += [norm_embeddings3]

                
            all_embeddings3 = tf.concat(all_embeddings3, 1)
            u1_embeddings, u2_embeddings = tf.split(all_embeddings3, [self.n_user, self.n_user], 0)
            # shape=(nums,dim)

            
            print("结束user_user_gcn")
        ua_e+=u1_embeddings


        if self.dataset!='music':
            self.adj_list2, self.adj_r_list2 = self._get_relational_adj_list2()

            lap_list2 = [self._si_norm_lap(adj) for adj in self.adj_list2]
            self.A2 = sum(lap_list2)

            self.A_fold_hat2 = self._split_A_hat(self.A2)
            ego_embeddings2 = tf.concat([self.all_weights['item_embed'], self.all_weights['entity_embed']], axis=0)
            all_embeddings2 = [ego_embeddings2]
            
            for k in range(0, self.n_layers-2):

                temp_embed2 = []
                for f in range(self.n_fold):
                    temp_embed2.append(tf.sparse_tensor_dense_matmul(self.A_fold_hat2[f], ego_embeddings2))

                side_embeddings2 = tf.concat(temp_embed2, 0)
                ego_embeddings2 = side_embeddings2
                all_embeddings2 += [ego_embeddings2]
            all_embeddings2 = tf.concat(all_embeddings2, 1)
            ii_embeddings, e_g_embeddings = tf.split(all_embeddings2, [self.n_item, self.n_entity], 0)

            ea_e=ea_embeddings+ e_g_embeddings
            print("结束item_entity_gcn")

        # (nums,dim) ATTENTION NET
        # ui_embeddings,ii_embeddings
        # [batch_size, dim, 1], [batch_size, 1, dim]增加维度
        v = tf.expand_dims(ui_embeddings, dim=2)# shape=(b,16,1)
        e = tf.expand_dims(ii_embeddings, dim=1)# shape=(b,1,16)

        # [batch_size, dim, dim]矩阵v*e
        c_matrix = tf.matmul(v, e)# shape=(b,16,16)
        attention_weights = tf.reduce_sum(e * c_matrix, axis=-1)
        attention_weights = tf.nn.softmax(attention_weights, dim=-1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
        user_embeddings = tf.reduce_sum(e * attention_weights_expanded, axis=1)

        # 0.89 0.90 0.92
        self.user_reference_adj = self._compute_reference_adj(ua_e, self.p)
        self.augmented_adj_1, self.augmented_features_1 = self._augment_graph(self.user_reference_adj, ua_e)
        self.item_reference_adj = self._compute_reference_adj(ea_e, self.p)
        self.augmented_adj_2, self.augmented_features_2 = self._augment_graph(self.item_reference_adj, ea_e)
        # print("self.user_reference_adj: ", self.user_reference_adj)

        self.Z_1 = self._gcn(self.augmented_adj_1, self.augmented_features_1)
        self.Z_2 = self._gcn(self.augmented_adj_2, self.augmented_features_2)
        return ua_e,user_embeddings, ea_e, lap_list,self.adj_r_list,self.A, self.Z_1, self.Z_2


class CrossCompressUnit(Layer):
    # 定义交叉压缩单元模型类
    def __init__(self, dim, name=None):
        #print('定义交叉压缩单元模型类')
        super(CrossCompressUnit, self).__init__(name)
        self.dim = dim
        with tf.variable_scope(self.name):
            # 在模型中开辟各自的空间，其中的变量均在这个空间内进行管理
            # get_variable创建新的tensorflow变量
            self.weight_vv = tf.get_variable(name='weight_vv', shape=(dim, 1), dtype=tf.float32)
            self.weight_ev = tf.get_variable(name='weight_ev', shape=(dim, 1), dtype=tf.float32)
            self.weight_ve = tf.get_variable(name='weight_ve', shape=(dim, 1), dtype=tf.float32)
            self.weight_ee = tf.get_variable(name='weight_ee', shape=(dim, 1), dtype=tf.float32)
            self.bias_v = tf.get_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee]


    def _call(self, inputs):
        #print('进行交叉单元运算')
        # [batch_size, dim]
        v, e = inputs# shape=(b,16)
     
        # [batch_size, dim, 1], [batch_size, 1, dim]增加维度
        v = tf.expand_dims(v, dim=2)# shape=(b,16,1)
        e = tf.expand_dims(e, dim=1)# shape=(b,1,16)

        # [batch_size, dim, dim]矩阵v*e
        c_matrix = tf.matmul(v, e)# shape=(b,16,16)

        # perm对矩阵进行转置，2*3*4变成2*4*3
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])# shape=(b,16,16)

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, self.dim])# shape=(b*16,16)
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])# shape=(b*16,16)

        # [batch_size, dim]  shape=(b,16)
        v_output = tf.reshape(tf.matmul(c_matrix, self.weight_vv) + tf.matmul(c_matrix_transpose, self.weight_ev),[-1, self.dim]) + self.bias_v
        e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_transpose, self.weight_ee),[-1, self.dim]) + self.bias_e

        # v_output = tf.reshape(tf.matmul(c_matrix, self.weight_vv),[-1, self.dim]) + self.bias_v
        # e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve),[-1, self.dim]) + self.bias_e

        return v_output, e_output


class Dense1(Layer):
    #                  8          8
    def __init__(self, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, name=None):
        #print('定义不带偏置的全连接层，将该全连接层作用于交叉后的特征向量，实现压缩的过程')
        super(Dense1, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name):
            #                                            shape=(8,8)
            self.weight = tf.get_variable(name='weight', shape=(input_dim, output_dim), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias', shape=output_dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight]

    def _call(self, inputs):
        # dropout的作用是丢弃一部分数据，防止模型过拟合
        #print('丢弃一部分数据，防止模型过拟合')
        # dropout中第二个参数设置神经元被选中的概率,参数 keep_prob: 表示的是保留的比例，
        # 假设为0.8 则 20% 的数据变为0，然后其他的数据乘以 1/keep_prob；keep_prob 越大，保留的越多；
        #                           self.dropout=0
        x = tf.nn.dropout(inputs, 1-self.dropout)
        output = tf.matmul(x, self.weight) + self.bias
        # tf.nn.relu用于用0替换负数
        return self.act(output)