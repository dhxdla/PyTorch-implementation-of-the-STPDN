from utils import normalization
import numpy as np
import warnings
warnings.filterwarnings("ignore")
class DataProcessor:
    '''
    这个类用于处理数据，将数据处理成合适的格式送入模型
    '''
    def __init__(self, data, input_length, output_length, train_ratio, valid_ratio, test_ratio, **kwargs):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio


    def time_in_day(self):
        TE = []
        length,node,_ = np.array(self.data).shape
        steps_per_day = int(24*60/5)
        TE = [i % steps_per_day for i in range(length)]
        TE = np.array(TE)
        TE = np.tile(TE, [1, node, 1]).transpose((2, 1, 0))
        return TE
    

    def day_in_week(self):
        TE = []
        length,node,_ = np.array(self.data).shape
        steps_per_day = int(24*60/5)

        TE = [(i // steps_per_day) % 7 for i in range(length)]
        TE = np.array(TE)
        TE = np.tile(TE, [1, node, 1])
        TE = TE.transpose((2, 1, 0))
        return TE


    def generate_data(self):
        data_week = self.day_in_week()
        data_day = self.time_in_day()
        head = 0
        input = []
        output = []
        tail = len(self.data)-self.output_length
        
        # data_train = data
        data_train, scalar = normalization(self.data,self.train_ratio)
        # 只对输入进行归一化，输出不进行归一化
        data_train = np.concatenate((data_train, data_day, data_week),axis=-1)

        while(head+self.input_length <= tail):
            #print(np.array(data[head-period*i-input_length:head-period*i]).shape)
            input.append(data_train[head: head+self.input_length])                      
            output.append(self.data[head+self.input_length: head+self.input_length+self.output_length])
            head += 1 
        print("数据格式生成成功，输入维度是：", np.array(input).shape,",", "输出维度是：", np.array(output).shape)
    
    
        length = len(output)
        train_idx = int(self.train_ratio*length)
        valid_idx = int((self.valid_ratio+self.train_ratio)*length)
        # print(length,train_idx,valid_idx)
        train = [input[:train_idx], output[:train_idx]]
        valid = [input[train_idx:valid_idx], output[train_idx:valid_idx]]
        test = [input[valid_idx:], output[valid_idx:]]
        print("切分数据成功，训练集维度是：", len(train[0]), "验证集维度是",len(valid[0]), "测试集维度是",len(test[0]))
        return train, valid, test, scalar

        
