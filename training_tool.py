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
import shutil
torch.manual_seed(1)

class Train:
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

    def train(self, train, valid, test, adj, scalar):
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
        node = np.array(train[0]).shape[-2]
        train_loader = list2loader(train, self.batch_size)
        valid_loader = list2loader(valid, self.batch_size)
        test_loader = list2loader(test, self.batch_size, shuffle = False)
        
        print("数据处理完成！！！")
        history = pd.DataFrame(columns = ["epoch","train_loss","test_loss","MAE","RMSE","MAPE"])

        model = STPAN(node, self.h, self.head, self.npb, self.in_len, self.ou_len, self.m, device, adj)
        for param_tuple in model.named_parameters():
            name, param = param_tuple
            if param.requires_grad:
                print("name = ", name)
                print("-" * 100)
        # summary(model, (64, 14, 12, 358, 1))
        # model = nn.DataParallel(model).cuda()   # 多卡训练时的写法
        # random_input = torch.rand(64, 14, 12, 358,1) # 随机一个 input
        # # 写入 tensorboard
        # writer = SummaryWriter()
        # writer.add_graph(model, random_input)
        # writer.close()
        # device = torch.device("cuda:0")
        val_metric_value = []
        test_metric_value = []
        model = model.to(device)
        adj = torch.tensor(adj).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        MMMMM = 10000
        for epoch in range(1, self.epoch+1):

            model.train()
            loss_sum = 0.0
            for step, (batch_x1, batch_y) in enumerate(train_loader):
                batch_y = batch_y.to(device)

                batch_x1 = batch_x1.to(device)


                # print(batch_x.size(),batch_y.size())
                # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                pre = model(batch_x1)

                pre = scalar.inverse_transform(pre)
                # loss_fun = nn.MSELoss()
                # print(pre.device,batch_y.device)
                # loss = loss_fun(pre, batch_y)
                loss = masked_mae_loss(pre, batch_y)
                if(torch.isnan(loss)):
                    print("loss")
                if(torch.isinf(loss)):
                    print("lossinf")
                
                # loss_fun = nn.L1Loss()
                # loss = loss_fun(pre,batch_y)
                # loss = nn.MSELoss(pre,batch_y)
                # loss = nn.SmoothL1Loss(pre,batch_y)
                loss.backward()
                optimizer.step()  
                loss_sum += loss.item()
                if step%100 == 0 and step != 0:
                    print(("step = %d || train_loss = %.4f")%(step,loss_sum/step))
                #print(/'Epoch: /', epoch, /'| Step: /', step, /'| batch x: /',batch_x.numpy(), /'| batch y: /', batch_y.numpy())
            train_step = step+1    
            torch.save(model.state_dict(), "./best_model/model_"+str(epoch)+".pt")


            model.eval()
            val_loss_sum = 0.0 
            MAE = 0.0
            MSE = 0.0
            MAPE = 0.0
            for step, (batch_x1, batch_y) in enumerate(valid_loader):
                # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                batch_y = batch_y.to(device)
                batch_x1 = batch_x1.to(device)


                with torch.no_grad():
                    pre = model(batch_x1)
                    pre = scalar.inverse_transform(pre)
                    loss = masked_mae_loss(pre, batch_y)
                    # loss_fun = nn.L1Loss()
                    # loss = loss_fun(pre,batch_y)
                    mae, mse, mape = evaluation(batch_y,pre, 0.0)
                    if MMMMM>mape:
                        MMMMM=mape
                        inddddd = epoch
                    # loss = nn.MSELoss(pre,batch_y)
                    # loss = nn.SmoothL1Loss(pre,batch_y)
                val_loss_sum += loss
                MAE += mae
                MSE += mse
                MAPE += mape
            val_loss_sum = val_loss_sum.cpu()
            step = step + 1
            RMSE = math.sqrt(MSE/step)
            info = (epoch, loss_sum/train_step, val_loss_sum/step, MAE/step, RMSE, MAPE/step)
            # history.loc[epoch-1] = info
            val_metric_value.append([MAE/step, RMSE, MAPE/step])
            print(("Epoch = %d, train_loss = %.4f, val_loss = %.4f, mae = %.4f, rmse = %.4f, mape = %.4f")%info)

            val_loss_sum = 0.0 
            MAE = 0.0
            MSE = 0.0
            MAPE = 0.0
            res_test = []
            for step, (batch_x1, batch_y) in enumerate(test_loader):
                # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                batch_y = batch_y.to(device)
                batch_x1 = batch_x1.to(device)


                with torch.no_grad():
                    pre = model(batch_x1)
                    pre = scalar.inverse_transform(pre)
                    loss = masked_mae_loss(pre, batch_y)
                    # loss_fun = nn.L1Loss()
                    # loss = loss_fun(pre,batch_y)
                    mae, mse, mape = evaluation(batch_y, pre, 0.0)
                    # loss = nn.MSELoss(pre,batch_y)
                    # loss = nn.SmoothL1Loss(pre,batch_y)
                    res_test.append(np.array(pre.cpu()))
                val_loss_sum += loss
                MAE += mae
                MSE += mse
                MAPE += mape

            val_loss_sum = val_loss_sum.cpu()
            step = step + 1
            RMSE = math.sqrt(MSE/step)
            info = (epoch, loss_sum/train_step, val_loss_sum/step, MAE/step, RMSE, MAPE/step)
            history.loc[epoch-1] = info
            test_metric_value.append([MAE/step, RMSE, MAPE/step])
            print(("Epoch = %d, train_loss = %.4f, test_loss = %.4f, mae = %.4f, rmse = %.4f, mape = %.4f")%info)

        
        history.to_csv("./history.csv")
            
        print("训练结束！")
        val_metric_value = np.array(val_metric_value)
        test_metric_value = np.array(test_metric_value)

        index = np.argmin(val_metric_value, axis=0)
        # min_mae = test_metric_value[index[0]]
        # min_rmse = test_metric_value[index[1]]
        min_mape = test_metric_value[index[2]]
        shutil.copyfile("./best_model/model_"+str(index[2]+1)+".pt", "./best_model/model.pt")
        print("前步中，最佳结果是：")
        # print("model_"+str(index[0]+1)+".pt", "model_"+str(index[1]+1)+".pt", "model_"+str(index[2]+1)+".pt")
        # print("以MAE为准：", min_mae)
        # print("以RMSE为准：", min_rmse)
        print("以MAPE为准：", min_mape)