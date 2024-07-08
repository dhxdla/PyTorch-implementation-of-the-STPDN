import numpy as np
import pandas as pd
import h5py
import yaml
import pickle
import scipy.sparse as sp

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

class GetArgs:
    def __init__(self):
        pass
    def get_yaml_data(self,yaml_file):
        # 打开yaml文件
        
        file = "config/"+yaml_file+".yaml"

        # print("***获取yaml文件数据***")
        f = open(file, 'r', encoding="utf-8")
        file_data = f.read()
        f.close()
        
        # print(file_data)
        # print("类型：", type(file_data))


        # 将字符串转化为字典或列表
        # print("***转化yaml数据为字典或列表***")
        data = yaml.load(file_data,Loader=yaml.FullLoader)
        # print(data)
        # print("类型：", type(data))
        return data['preprocess'], data['model'], data['train']

    # data_config = config['preprocess']
    # model_config = config['model']
    # print(data_config)

class DataLoader:
    """
    这个类的作用是读取不同数据集，包括pems03/04/07/08/bay/metr-la
    """
    def __init__(self,path,data_name,**kwargs):
        self.path = path
        self.data_name = data_name


    def read(self):
        if(self.data_name=='pems04' or self.data_name=='pems07' or self.data_name=='pems08'):
            #数据的维度是(26208, 358, 1)，共26208条数据，358个结点，1个交通流量
            data = np.load(self.path)['data']
            data = np.expand_dims(data[:,:,0], axis=-1)
            graph = pd.read_csv(self.path[:-4]+'.csv')
            node = data.shape[1]
            adj = 0
            adj = np.zeros((node,node), dtype=float)
            for i in range(node):
                adj[i][i]=1
            for _, row in graph.iterrows():
                from_ = int(row['from'])
                to_ = int(row['to'])
                adj[from_][to_] = 1
                adj[to_][from_] = 1
            print("数据读取成功，数据维度是：",data.shape)
            return data, adj
        
        elif self.data_name=='pems03':
            data = np.load(self.path)['data']
            data = np.expand_dims(data[:,:,0], axis=-1)
            graph = pd.read_csv(self.path[:-4]+'.csv')
            num_node = list(set(graph['from'].tolist() + graph['to'].tolist()))
            node = data.shape[1]
            adj = 0
            adj = np.zeros((node,node), dtype=float)
            for i in range(node):
                adj[i][i]=1
            for _, row in graph.iterrows():
                a = int(row['from'])
                b = int(row['to'])
                from_ = num_node.index(a)
                to_ = num_node.index(b)
                adj[from_][to_] = 1
                adj[to_][from_] = 1
            print("数据读取成功，数据维度是：",data.shape)
            return data, adj
        
        elif(self.data_name=='pems-bay'):
            data=h5py.File(self.path+"pems-bay.h5","r+")
            data = np.array(data["speed"]["block0_values"])
            data = np.expand_dims(data, axis=-1)
            _,_, adj = load_adj(self.path+"adj_mx_bay.pkl", "transition")
            adj = np.array(adj)
            print("数据读取成功，数据维度是：",data.shape)
            return data, adj
        elif(self.data_name=='metr-la'):
            data = pd.read_hdf(self.path+"metr-la.h5")
            data = np.expand_dims(data, axis=-1)
            _,_, adj = load_adj(self.path+"adj_mx_la.pkl", "transition")
            adj = np.array(adj)
            print("数据读取成功，数据维度是：",data.shape)
            return data, adj
        elif(self.data_name=='nycbike1'):
            data = np.load(self.path+"data.npy")
            data = data[...,0]
            data = np.expand_dims(data, axis=-1)
            adj = np.load(self.path+"adj_mx.npz")['adj_mx']
            print("数据读取成功，数据维度是：",data.shape)
            return data, adj
            # data1 = np.load(self.path+"train.npz")
            # data2 = np.load(self.path+"val.npz")
            # data3 = np.load(self.path+"test.npz")
            # # print(data1['x'].shape, data1['y'].shape,data2['x'].shape,data3['x'].shape)
            # train = []
            # val = []
            # test = []
            # for i in range(len(data1['x'])):
            #     if i % 19 == 0:
            #         train.append(data1['x'][i])
            # for i in range(len(data2['x'])):
            #     if i % 19 == 0:
            #         val.append(data2['x'][i])
            # for i in range(len(data3['x'])):
            #     if i % 19 == 0:
            #         test.append(data3['x'][i])
            # train = np.concatenate(train,axis=0)
            # val = np.concatenate(val,axis=0)
            # test = np.concatenate(test,axis=0)

            # X = np.concatenate((train, val, test),axis=0)
            # np.save(self.path+'data',X)
            # print(X.shape)
            # adj = np.load(self.path+"adj_mx.npz")['adj_mx']

            # return [[data1['x'],data1['y']],[data2['x'],data2['y']],[data3['x'],data3['y']]], adj
            # # data = data1+data2+data3

        else:
            print("未提前处理过该数据")