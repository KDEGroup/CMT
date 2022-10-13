import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

graph_data = np.load("/home/zqxu/MHTGNN/data/phase1_gdata.npz")
x = graph_data['x']
y = graph_data['y']
edge_type = graph_data['edge_type']
edge_index = graph_data['edge_index']
edge_timestamp = graph_data['edge_timestamp']
train_mask = graph_data['train_mask']
test_mask = graph_data['test_mask']

edges = np.concatenate((edge_index, np.expand_dims(edge_type, axis=1)), axis=1)
edges = np.concatenate((edges, np.expand_dims(edge_timestamp, axis=1)), axis=1)
edges = pd.DataFrame(edges, columns=['src_id', 'dst_id', 'type', 'ts'])
edges.sort_values(by=['ts'], inplace=True)
max_ts = max(edges['ts'])
edges_ts_cnt = [0] * max_ts
for i in tqdm(range(1, max_ts+1)):
    edges_ts_cnt[i-1] = edges[edges['ts'] <= i].count()['src_id']
with open("/home/zqxu/MHTGNN/code/edges_ts_cnt.pkl", "wb") as f:
    pickle.dump(edges_ts_cnt, f)