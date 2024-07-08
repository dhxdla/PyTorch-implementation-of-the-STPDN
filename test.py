import torch
import torch.utils.data as Data
import torch.nn as nn
import datetime
from utils import evaluation
from torch.utils.tensorboard import SummaryWriter
from stpan import STPAN
import numpy as np
from utils import list2loader,masked_mae_loss
import pandas as pd
from matplotlib import pyplot as plt
import os
import math
torch.manual_seed(1)

class Test:
    def __init__(self, batch_size, epoch, learning_rate, hidden, head, num_Period_Block, input_length, output_length, m, device, **kwargs):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = learning_rate
        self.h = hidden
        self.head = head
        self.npb = num_Period_Block
        self.in_len = input_length
        self.ou_len = output_length
        self.m = m
        self.device = device

    def test(self, test, adj, scalar):
        device = "cuda:" + str(self.device)
        adj = torch.tensor(adj,dtype=torch.float32).to(device)
        print("cuda可用吗:", torch.cuda.is_available())
        # train_input = torch.tensor(np.array(data[0]))
        # train_output = torch.tensor(np.array(data[1]))
        # train_torch_data = Data.TensorDataset(train_input, train_output)
        # train_loader = Data.DataLoader(dataset=train_torch_data,
        #                          batch_size=self.batch_size,
        #                          shuffle=True,
        #                          num_workers=2)
        node = np.array(test[0]).shape[-2]

        test_loader = list2loader(test, self.batch_size, shuffle = False)
        
        print("数据处理完成！！！")


        model = STPAN(node, self.h, self.head, self.npb, self.in_len, self.ou_len, self.m, device, adj)
        model.load_state_dict(torch.load('./best_model/model.pt'))
        model.eval()
        # summary(model, (64, 14, 12, 358, 1))GMN/best_model/model_28.pt
        # model = nn.DataParallel(model).cuda()   # 多卡训练时的写法
        # random_input = torch.rand(64, 14, 12, 358,1) # 随机一个 input
        # # 写入 tensorboard
        # writer = SummaryWriter()
        # writer.add_graph(model, random_input)
        # writer.close()
        # device = torch.device("cuda:0")


        model = model.to(device)
        adj = torch.tensor(adj).to(device)


        MAE = 0.0
        MSE = 0.0
        MAPE = 0.0
        A = []
        B = []
        C = []
        Pre = []
        Label = []

        res_test = []
        for step, (batch_x1, batch_y) in enumerate(test_loader):
            # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            Label.append(batch_y)
            batch_y = batch_y.to(device)
            batch_x1 = batch_x1.to(device)
            print(step)

            with torch.no_grad():
                pre= model(batch_x1)
                pre = scalar.inverse_transform(pre)
                # r = scalar.inverse_transform(r)
                # p = scalar.inverse_transform(p)
                Pre.append(pre.cpu())
                


                # loss_fun = nn.L1Loss()
                # loss = loss_fun(pre,batch_y)
                mae, mse, mape = evaluation(batch_y, pre, 0.0)
                temp_a = []
                temp_b = []
                temp_c = []
                for t in range(12):
                    a,b,c = evaluation(batch_y[:,t,...], pre[:,t,...], 0.0)
                    temp_a.append(a)
                    temp_b.append(b)
                    temp_c.append(c)
                
                # loss = nn.MSELoss(pre,batch_y)
                # loss = nn.SmoothL1Loss(pre,batch_y)
                res_test.append(np.array(pre.cpu()))

            MAE += mae
            MSE += mse
            MAPE += mape
            A.append(temp_a)
            B.append(temp_b)
            C.append(temp_c)

        # res_test = np.array(res_test).transpose(0,1,3,2)
        # # res_test.tofile('./history/{}-testpre.dat'.format(epoch))
        # real_test = np.array(test[2]).squeeze(-1).transpose(0,2,1)
        # for k in range(10):
        #     plt.figure()
        #     plt.plot(res_test[0,0,k,:])
        #     plt.plot(real_test[0,k,:])
        #     plt.savefig('./pic/{}-{}.png'.format(epoch, k + 1))
        A = np.sum(np.array(A),axis=0)
        B = np.sum(np.array(B),axis=0)
        C = np.sum(np.array(C),axis=0)


        step = step + 1
        RMSE = math.sqrt(MSE/step)
        info = (MAE/step, RMSE, MAPE/step)

        STEP12_MAE = [i/step for i in A]
        STEP12_RMSE = [math.sqrt(i/step) for i in B]
        STEP12_MAPE = [i/step for i in C]


        print(("mae = %.4f, rmse = %.4f, mape = %.4f")%info)
        print("单步预测")
        print(STEP12_MAE)
        print(STEP12_MAPE)
        print(STEP12_RMSE)
        np.save("PRE.npy", Pre)
        np.save("Label.npy", Label)


