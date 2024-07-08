import argparse
from read_data import DataLoader,GetArgs
from data_processor import DataProcessor
from training_tool import Train
from test import Test

#设置可选的参数
parser = argparse.ArgumentParser()
parser.add_argument('-file',type=str,help='Path of configuration file')

#file是字符串类型的，用于选择配置文件
file = parser.parse_args().file

#读取配置文件，返回配置文件中的每一项
get_args = GetArgs()
preprocess_args, model_args, train_args = get_args.get_yaml_data(file)
print(preprocess_args)
print(model_args)
print(train_args)

#读取原始数据
data_loader = DataLoader(**preprocess_args)
data, adj = data_loader.read()

#处理数据
processor = DataProcessor(data, **preprocess_args)
train_data, valid_data, test_data, scalar = processor.generate_data()

#开始训练
if preprocess_args['mode'] == "train":
    trainer = Train(**train_args, **model_args)
    trainer.train(train_data, valid_data, test_data, adj, scalar)
else:
    tester = Test(**train_args, **model_args)
    tester.test(test_data, adj, scalar)


