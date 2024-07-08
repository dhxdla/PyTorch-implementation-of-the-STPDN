import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import pandas as pd



class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def normalization(data,ratio):
    # print(np.array(self.data.squeeze().mean()).shape)
    l = len(data)
    num = int(l * ratio)
    mean=np.array(data[:num]).squeeze().mean()
    std=np.array(data[:num]).squeeze().std() #[..., 0]
    scalar = StandardScaler(mean,std)
    data = scalar.transform(data)
    print("标准化成功，均值是：", mean, "方差是：", std)
    return data, scalar

class StandardScaler2:
    def __init__(self,input,output):
        self.i = input
        self.o = output
    def transform(self):
        # scalar = []
        # inp = self.i
        # outp = self.o
        # for i in range(len(self.i)):
        #     s = self.i[i][-1][-1]
        #     scalar.append(s)
        #     inp[i] = self.i[i]/s
        #     outp[i] = self.o[i]/s
        # return inp,outp,scalar
        pass
    def inverse_transform(self):
        pass
    

def list2loader(data, batch_size, shuffle=True):
    input = torch.tensor(np.array(data[0]),dtype=torch.float32)
    # input2 = torch.tensor(np.array(data[1]),dtype=torch.float32)
    output = torch.tensor(np.array(data[1]),dtype=torch.float32).squeeze(-1)

    torch_data = Data.TensorDataset(input, output)
    loader = Data.DataLoader(dataset=torch_data,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=True,
                                num_workers=2)
    return loader




def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # loss[loss != loss] = 0
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = loss.mean()
    return loss


 
def evaluation(labels, preds, null_val=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    mape = torch.mean(loss)
    mape = mape.cpu()
    # mape = mape.mean().cpu()

    mae = torch.abs(labels - preds)
    mae = mae * mask
    # mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    mae = mae.mean().cpu()

    mse = torch.pow(labels - preds, 2)
    mse = mse * mask
    # rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    mse = mse.mean().cpu()
    return mae, mse, mape #mse


