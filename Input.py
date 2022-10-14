from tkinter import X
import numpy as np
import dgl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import math
import pickle
import random
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def loadXinyeDataHomo():
    graph_data = np.load("XXX/phase1_gdata.npz")
    x = graph_data['x']
    # x[x==-1] = np.nan
    # col_mean = np.nanmean(x, axis=0)
    # inds = np.where(np.isnan(x))
    # x[inds] = np.take(col_mean, inds[1])
    # x = x / np.linalg.norm(x)
    y = graph_data['y']
    label = torch.FloatTensor(y)
    edge_index = graph_data['edge_index']
    train_mask = graph_data['train_mask']
    n_homo_features = x.shape[1]

    src, dst = torch.tensor(edge_index[:, 0]), torch.tensor(edge_index[:, 1])
    graph = dgl.graph((src, dst), num_nodes=x.shape[0])
    graph = dgl.to_bidirected(graph)
    graph = dgl.add_self_loop(graph)
    graph.ndata['feature'] = torch.FloatTensor(x)

    with open("/XXX/u_train_test_Xinye.pickle", "rb") as fp:
        X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)

    train_user_idx = X_train_p + X_train_n
    test_user_idx = X_test_p + X_test_n

    random.shuffle(train_user_idx)
    random.shuffle(test_user_idx)

    test_uid = set(test_user_idx)
    edges = pd.DataFrame(edge_index, columns=['src', 'dst'])
    edges_train = edges[~edges['src'].isin(test_uid) & ~edges['dst'].isin(test_uid)].to_numpy()
    src, dst = torch.tensor(edges_train[:, 0]), torch.tensor(edges_train[:, 1])
    graph_train = dgl.graph((src, dst), num_nodes=x.shape[0])
    graph_train = dgl.to_bidirected(graph_train)
    graph_train = dgl.add_self_loop(graph_train)
    graph_train.ndata['feature'] = torch.FloatTensor(x)

    return [graph_train, graph], n_homo_features, train_user_idx, test_user_idx, label

def loadXinyeDataHetero():
    graph_data = np.load("XXX/phase1_gdata.npz")
    x = graph_data['x']
    # x[x==-1] = np.nan
    # col_mean = np.nanmean(x, axis=0)
    # inds = np.where(np.isnan(x))
    # x[inds] = np.take(col_mean, inds[1])
    # x = x / np.linalg.norm(x)
    y = graph_data['y']
    label = torch.FloatTensor(y)
    edge_type = graph_data['edge_type']
    edge_type[edge_type < 4] = 1
    edge_type[(edge_type >= 4)&(edge_type < 8)] = 2
    edge_type[(edge_type >= 8)&(edge_type <= 11)] = 3
    edge_index = graph_data['edge_index']
    train_mask = graph_data['train_mask']
    n_hetero_features = x.shape[1]

    edges = np.concatenate((edge_index, np.expand_dims(edge_type, axis=1)), axis=1)
    edges = pd.DataFrame(edges, columns=['src', 'dst', 'type'])
    edges_one = edges[edges['type'] == 1].drop(columns=['type']).to_numpy()
    edges_two = edges[edges['type'] == 2].drop(columns=['type']).to_numpy()
    edges_three = edges[edges['type'] == 3].drop(columns=['type']).to_numpy()
    one_src, one_dst = torch.tensor(edges_one[:, 0]), torch.tensor(edges_one[:, 1])
    two_src, two_dst = torch.tensor(edges_two[:, 0]), torch.tensor(edges_two[:, 1])
    three_src, three_dst = torch.tensor(edges_three[:, 0]), torch.tensor(edges_three[:, 1])
    
    hg = dgl.heterograph(
        data_dict={
            ('user', 'one', 'user'): (one_src, one_dst),
            ('user', 'one_by', 'user'): (one_dst, one_src),
            ('user', 'two', 'user'): (two_src, two_dst),
            ('user', 'two_by', 'user'): (two_dst, two_src),
            ('user', 'three', 'user'): (three_src, three_dst),
            ('user', 'three_by', 'user'): (three_dst, three_src)
        },
        num_nodes_dict={
            'user': x.shape[0]
        }
    )
    hg.nodes['user'].data['feature'] = torch.FloatTensor(x)

    train_user_idx, test_user_idx = train_test_split(train_mask, test_size=0.2, random_state=42)
    test_uid = set(test_user_idx)
    edges_train = edges[~edges['src'].isin(test_uid) & ~edges['dst'].isin(test_uid)]
    edges_one = edges_train[edges_train['type'] == 1].drop(columns=['type']).to_numpy()
    edges_two = edges_train[edges_train['type'] == 2].drop(columns=['type']).to_numpy()
    edges_three = edges_train[edges_train['type'] == 3].drop(columns=['type']).to_numpy()
    one_src, one_dst = torch.tensor(edges_one[:, 0]), torch.tensor(edges_one[:, 1])
    two_src, two_dst = torch.tensor(edges_two[:, 0]), torch.tensor(edges_two[:, 1])
    three_src, three_dst = torch.tensor(edges_three[:, 0]), torch.tensor(edges_three[:, 1])

    hg_train = dgl.heterograph(
        data_dict={
            ('user', 'one', 'user'): (one_src, one_dst),
            ('user', 'one_by', 'user'): (one_dst, one_src),
            ('user', 'two', 'user'): (two_src, two_dst),
            ('user', 'two_by', 'user'): (two_dst, two_src),
            ('user', 'three', 'user'): (three_src, three_dst),
            ('user', 'three_by', 'user'): (three_dst, three_src)
        },
        num_nodes_dict={
            'user': x.shape[0]
        }
    )
    hg_train.nodes['user'].data['feature'] = torch.FloatTensor(x)

    num_user_nodes = x.shape[0]
    fraud_user = set(np.where(y==1)[0].tolist())
    normal_user = set(np.where(y==0)[0].tolist())

    with open("XXX/u_train_test_Xinye.pickle", "rb") as fp:
        X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)

    train_user_idx = X_train_p + X_train_n
    test_user_idx = X_test_p + X_test_n

    random.shuffle(train_user_idx)
    random.shuffle(test_user_idx)
    # X_train = set(X_train_p + X_train_n)
    # X_test = set(X_test_p + X_test_n)

    # train_user_idx = set(train_user_idx)
    # test_user_idx = set(test_user_idx)

    # print(len(X_train & train_user_idx) == len(X_train))

    # X_train_p = list(fraud_user & train_user_idx)
    # X_train_n = list(normal_user & train_user_idx)
    # X_test_p = list(fraud_user & test_user_idx)
    # X_test_n = list(normal_user & test_user_idx)

    # with open("/home/zqxu/MHTGNN/data/u_train_test_Xinye.pickle", "wb") as fp:
    #     pickle.dump((X_train_p, X_train_n, X_test_p, X_test_n), fp)

    return [hg_train, hg], n_hetero_features, train_user_idx, test_user_idx, label

def loadXinyeDataHomoDynamic():
    graph_data = np.load("XXX/phase1_gdata.npz")
    edge_index = graph_data['edge_index']
    edge_timestamp = graph_data['edge_timestamp']
    gap = math.ceil(max(edge_timestamp) / 7)
    for i in range(1, 8):
        edge_timestamp[(edge_timestamp >= (i-1)*gap)&(edge_timestamp < i*gap)] = i

    with open("XXX/u_train_test_Xinye.pickle", "rb") as fp:
        X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)

    train_user_idx = X_train_p + X_train_n
    test_user_idx = X_test_p + X_test_n

    edges = np.concatenate((edge_index, np.expand_dims(edge_timestamp, axis=1)), axis=1)
    edges = pd.DataFrame(edges, columns=['src', 'dst', 'ts'])
    edges.sort_values(by=['ts'], inplace=True)

    return train_user_idx, test_user_idx, edges

def loadXinyeDataHeteroDynamic():
    graph_data = np.load("XXX/phase1_gdata.npz")
    x = graph_data['x']
    y = graph_data['y']
    label = torch.FloatTensor(y)
    edge_type = graph_data['edge_type']
    edge_type[edge_type < 4] = 1
    edge_type[(edge_type >= 4)&(edge_type < 8)] = 2
    edge_type[(edge_type >= 8)&(edge_type <= 11)] = 3
    edge_index = graph_data['edge_index']
    edge_timestamp = graph_data['edge_timestamp']
    gap = math.ceil(max(edge_timestamp) / 7)
    for i in range(1, 8):
        edge_timestamp[(edge_timestamp >= (i-1)*gap)&(edge_timestamp < i*gap)] = i
    n_hetero_features = x.shape[1]

    with open("XXX/u_train_test_Xinye.pickle", "rb") as fp:
        X_train_p, X_train_n, X_test_p, X_test_n = pickle.load(fp)

    train_user_idx = X_train_p + X_train_n
    test_user_idx = X_test_p + X_test_n

    edges = np.concatenate((edge_index, np.expand_dims(edge_type, axis=1)), axis=1)
    edges = np.concatenate((edges, np.expand_dims(edge_timestamp, axis=1)), axis=1)
    edges = pd.DataFrame(edges, columns=['src', 'dst', 'type', 'ts'])
    edges.sort_values(by=['ts'], inplace=True)
    edges_one = edges[edges['type'] == 1].drop(columns=['type', 'ts']).to_numpy()
    edges_two = edges[edges['type'] == 2].drop(columns=['type', 'ts']).to_numpy()
    edges_three = edges[edges['type'] == 3].drop(columns=['type', 'ts']).to_numpy()

    one_src, one_dst = torch.tensor(edges_one[:, 0]), torch.tensor(edges_one[:, 1])
    two_src, two_dst = torch.tensor(edges_two[:, 0]), torch.tensor(edges_two[:, 1])
    three_src, three_dst = torch.tensor(edges_three[:, 0]), torch.tensor(edges_three[:, 1])
    
    hg = dgl.heterograph(
        data_dict={
            ('user', 'one', 'user'): (one_src, one_dst),
            ('user', 'one_by', 'user'): (one_dst, one_src),
            ('user', 'two', 'user'): (two_src, two_dst),
            ('user', 'two_by', 'user'): (two_dst, two_src),
            ('user', 'three', 'user'): (three_src, three_dst),
            ('user', 'three_by', 'user'): (three_dst, three_src)
        },
        num_nodes_dict={
            'user': x.shape[0]
        }
    )
    hg.nodes['user'].data['feature'] = torch.FloatTensor(x)

    return hg, x, n_hetero_features, train_user_idx, test_user_idx, edges, label

def user_seq_construct():
    graph_data = np.load("XXX/phase1_gdata.npz")
    edge_index = graph_data['edge_index']
    edge_timestamp = graph_data['edge_timestamp']
    gap = math.ceil(max(edge_timestamp) / 14)
    for i in range(1, 15):
        edge_timestamp[(edge_timestamp >= (i-1)*gap)&(edge_timestamp < i*gap)] = i-1
    edges = np.concatenate((edge_index, np.expand_dims(edge_timestamp, axis=1)), axis=1)
    x = graph_data['x']
    time_span = 14
    user_interaction_record = pd.DataFrame(edges, columns=['uid', 'fid', 'action_ts'])
    num_user_nodes = x.shape[0]
    #  根据时间顺序对用户交互行为排序
    user_interaction_record.sort_values(by=['action_ts'], inplace=True)
    #  每个用户，在每个时间点下交互过的用户
    user_seq = [[[] for i in range(time_span)] for i in range(num_user_nodes)]
    for index, rows in tqdm(user_interaction_record.iterrows(), total=user_interaction_record.shape[0]):
        uid = rows['uid']
        fid = rows['fid']
        t = int(rows['action_ts'])
        user_seq[uid][t].append(fid)
    #  去除每个用户中不存在交互记录的时间点
    for i in tqdm(range(len(user_seq))):
        user_seq[i] = [x for x in user_seq[i] if len(x) > 0]
    with open('XXX/user_relation_seq.pickle', 'wb') as f:
        pickle.dump(user_seq, f)    
    #  每个用户出现在其他用户的交互序列中的位置, 以及对应的子序列的长度
    #  例子：1->[6,2,5,7,3,19], 则user_appear_location[5]={"0": {2, 4}}
    user_appear_location = [{} for i in range(num_user_nodes)]
    for i in tqdm(range(len(user_seq))):
        seq = user_seq[i]
        if len(seq) > 0:
            for t in range(len(seq)):
                #  users为用户i在t时刻交互过的用户
                users = seq[t]
                for user in users:
                    # t->[0, len(seq)-1]
                    user_appear_location[user][i] = [t, len(seq)-t]
    with open('XXX/user_appear_location.pickle', 'wb') as f:
        pickle.dump(user_appear_location, f)

    sample_num = 1
    sampled_user_seq = [[] for i in range(num_user_nodes)]
    for i in tqdm(range(len(user_seq))):
        #  第i个用户为起始顶点的用户序列
        seq = user_seq[i]
        if len(seq) > 0:
            constructed_seq = []
            # sample_seq = []
            # dfs(constructed_seq, sample_seq, seq, 0, 0, sample_num, len(seq))
            sample_sequence(constructed_seq=constructed_seq, seq=seq, offset=0, sample_num=sample_num)
            sampled_user_seq[i] = constructed_seq
    with open('XXX/sampled_user_seq.pickle', 'wb') as f:
        pickle.dump(sampled_user_seq, f)

    for i in tqdm(range(len(sampled_user_seq))):
        #  不存在以第i个用户为起始顶点的序列
        if(len(sampled_user_seq[i]) == 0):
            appear_location = user_appear_location[i]
            #  appear_location记录了用户i在其他用户序列的出现位置及子序列长度
            #  例: {"0": {2, 4}}
            appear_location = dict(sorted(appear_location.items(), key=lambda x: x[1][1]), reverse=True)
            base_user = list(appear_location.keys())[0]
            start_position = appear_location[base_user][0]
            seq_len = appear_location[base_user][1]
            constructed_seq = []
            sample_seq = []
            seq = user_seq[base_user]
            # dfs(constructed_seq, sample_seq, seq, offset=start_position+1, i=start_position+1, sample_num=sample_num, sample_len=seq_len)
            sample_sequence(constructed_seq=constructed_seq, seq=seq, offset=start_position+1, sample_num=sample_num)
            sampled_user_seq[i] = constructed_seq
    with open('XXX/sampled_user_seq_complete.pickle', 'wb') as f:
        pickle.dump(sampled_user_seq, f) 

    with open('XXX/sampled_user_seq_complete.pickle', 'rb') as f:
        transformed_user_seq = pickle.load(f)
    uf = torch.FloatTensor(x)
    num_nodes = len(transformed_user_seq)
    assert len(transformed_user_seq) == uf.shape[0]
    user_constructed_seq_input = [[] for i in range(num_nodes)]
    for i in tqdm(range(len(transformed_user_seq))):
        user_seqs = transformed_user_seq[i]
        for seq in user_seqs:
            if seq:
                input = torch.index_select(uf, 0, torch.tensor(seq))
            # else:
            #     input = torch.zeros(1, 17)
                user_constructed_seq_input[i].append(input)
    with open('XXX/user_relation_seq_input.pickle', 'wb') as f:
        pickle.dump(user_constructed_seq_input, f) 

def sample_sequence(constructed_seq, seq, offset, sample_num):
    for _ in range(sample_num):
        sample_seq = []
        for i in range(offset, len(seq)):
            e = random.choice(seq[i])
            sample_seq.append(e)
        constructed_seq.append(sample_seq)

def dfs(constructed_seq, sample_seq, seq, offset, i, sample_num, sample_len):
    if i-offset == sample_len:
        constructed_seq.append(sample_seq.copy())
    else:
        for e in seq[i]:
            sample_seq.append(e)
            if len(constructed_seq) < sample_num:
                dfs(constructed_seq, sample_seq, seq, offset, i+1, sample_num, sample_len)
            sample_seq.pop(len(sample_seq) - 1)

def node_augment_construct():
    path = "XXX/data/"
    with open(path + 'user_relation_seq.pickle', 'rb') as f:
        user_seq = pickle.load(f)
    
    node_augment_hash = {}
    for i in range(len(user_seq)):
        seq = user_seq[i]
        if i % 10000 == 0:
            print("process finished: %d" % i)
        if len(seq) > 0:
            for t in range(len(seq)):
                seq_t = seq[t]
                for j in range(len(seq_t)):
                    if seq_t[j] not in node_augment_hash:
                        node_augment_hash[seq_t[j]] = {}
                        if i not in node_augment_hash[seq_t[j]]:
                            augment_list = []
                            for k in range(j+1, len(seq)):
                                augment_list.extend(seq[k])
                        else:
                            augment_list = node_augment_hash[seq_t[j]][i]
                            for k in range(j+1, len(seq)):
                                augment_list.extend(seq[k])
                        if augment_list:
                            augment_unique_list = list(set(augment_list))
                            node_augment_hash[seq_t[j]][i] = augment_unique_list
    
    with open(path + 'node_augment_hash_with_hetero.pickle', 'wb') as f:
        pickle.dump(node_augment_hash, f)

def edges_examine():
    graph_data = np.load("XXX/phase1_gdata.npz")
    x = graph_data['x']
    edge_index = graph_data['edge_index']
    edge_index = pd.DataFrame(edge_index, columns=['src', 'dst'])
    edge_uid = edge_index['src'].values.tolist() + edge_index['dst'].values.tolist()
    edge_uid = set(edge_uid)
    ids = [i for i in range(x.shape[0])]
    ids = set(ids)
    print(len(ids & edge_uid) == len(ids))

if __name__ == '__main__':
    user_seq_construct()