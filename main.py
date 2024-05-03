import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from utils import *
from model import Net
# 对模型参数进行随机初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.uniform_(m.weight, -0.1, 0.1)  # 使用均匀分布进行初始化，范围为[-0.1, 0.1]
        # nn.init.zeros_(m.bias)  # 将偏置初始化为0

if __name__ == '__main__':
    # 参数设置
    seq_length = 3   # 时间步长
    input_size = 1  # 原本为3，现在为5， 删去postcode与time
    num_layers = 1 #  2  4
    hidden_size = 256  #128 # 512??
    batch_size = 32
    n_iters = 10*10000 # 50000 5000
    lr = 2*1e-6  #2*1e-6     #0.001
    output_size = 1
    split_ratio = 0.9
    path = './datasets/all_stocks_2006-01-01_to_2018-01-01.csv'
    moudle = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(moudle.parameters(), lr=lr)
    scaler = MinMaxScaler()
    print(moudle)
    model_list = []
    optimizer_list = []
    tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    train = False
    evaluate = True

    df = pd.read_csv(path)
    AAPL = df.loc[df['Name'] == tech_list[0]]
    GOOGL = df.loc[df['Name'] == tech_list[1]]
    MSFT = df.loc[df['Name'] == tech_list[2]]
    AMZN = df.loc[df['Name'] == tech_list[3]]
    company_list = [AAPL, GOOGL, MSFT, AMZN]
    '''
    若要针对原数据进行聚合，可使用下面注释了的代码
    '''
    # for i in range(len(company_list)):
    #     company_list[i]['Date'] = pd.to_datetime(company_list[i]['Date'])
    #     company_list[i] = company_list[i].resample('M', on='Date')['Close'].median().to_frame()
    #     company_list[i].reset_index(inplace=True)
    #     company_list[i] = company_list[i][['Date','Close']]
    for i in range(len(tech_list)):
        temp_model = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr)
        model_list.append(temp_model)
        optimizer_list.append(temp_optimizer)
    data = [company.filter(['Close'])  for company in company_list]   # (4,seq_length) type:DataFrame
    dataset = [data[i].values for i in range(len(data))]  # (4,seq_length) type:List
    x_train_list = []
    y_train_list = []
    scaler = [MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(-1, 1)),
              MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(-1, 1))]
    for i in range(len(dataset)):
        temp = dataset[i]  # (data_length, 1)
        normalied_temp = scaler[i].fit_transform(temp)
        # create windows....
        x, y = split_windows(normalied_temp, seq_length)
        x_train_list.append(x)
        y_train_list.append(y)
    iter = 0
    total_loss = []
    if train:
        for i in range(len(x_train_list)):
            loss_list = []
            x_data, y_data, x_train, y_train, x_test, y_test = split_data(x_train_list[i], y_train_list[i], split_ratio)
            train_loader, test_loader, num_epochs = data_generator(x_train, y_train, x_test, y_test, n_iters=n_iters,
                                                                   batch_size=batch_size)
            model_list[i].cuda()
            moudle = model_list[i]
            moudle.apply(init_weights)
            moudle.train()

            optimizer = optimizer_list[i]
            for epochs in range(num_epochs):
                for number, (batch_x, batch_y) in enumerate(train_loader):
                    outputs = moudle(batch_x)
                    optimizer.zero_grad()  # 将每次传播时的梯度累积清除
                    loss = criterion(outputs, batch_y)  # 计算损失
                    loss.backward()  # 反向传播
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0:
                        print("iter: %d, loss: %1.5f" % (iter, loss.item()))
                        loss_list.append(loss.item())
            total_loss.append(loss_list)
            moudle.eval()
            predict = moudle(x_data).cpu().detach().numpy().reshape(-1, 1)
            predict = scaler[i].inverse_transform(predict).reshape(-1)
            plt.subplot(2, 2, i + 1)
            plt.plot(pd.to_datetime(company_list[i]['Date']).values, company_list[i]['Close'].values)
            plt.plot(pd.to_datetime(company_list[i]['Date']).values[:-(seq_length + 1)], predict)
            plt.ylabel('Close Price USD ($)', fontsize=18)
            plt.xlabel('Date', fontsize=18)
            plt.title(f"Close price for {tech_list[i]}")
        plt.tight_layout()
        plt.show()
        for i in range(len(total_loss)):
            plt.subplot(2, 2, i + 1)
            plt.plot(total_loss[i])
            plt.ylabel('Loss Value', fontsize=18)
            plt.xlabel('Iter', fontsize=18)
            plt.title(f"Loos for {tech_list[i]}")
        plt.tight_layout()
        plt.show()
    '''
    保存模型
    '''
    if train:
        for i in range(len(tech_list)):
            torch.save({'model': model_list[i].state_dict()}, f"state_dict_of_{tech_list[i]}.pth")
            torch.save(model_list[i], f"model_of_{tech_list[i]}.pth")

    '''
    模型评估
    '''
    if evaluate:
        for i in range(len(tech_list)):
            x_data, y_data, x_train, y_train, x_test, y_test = split_data(x_train_list[i], y_train_list[i], split_ratio)
            state_dict = torch.load(f"state_dict_of_{tech_list[i]}.pth")
            model_list[i].load_state_dict(state_dict['model'])
            moudle = model_list[i]
            moudle.cuda()
            moudle.eval()
            predict = moudle(x_test).cpu().detach().numpy().reshape(-1, 1)
            predict = scaler[i].inverse_transform(predict).reshape(-1)
            plt.subplot(2, 2, i + 1)
            plt.plot(scaler[i].inverse_transform(y_test.cpu().reshape(-1, 1)))
            plt.plot(predict)
            plt.ylabel('Value', fontsize=18)
            plt.xlabel('Point', fontsize=18)
            plt.title(f"Predict {tech_list[i]}")
        plt.tight_layout()
        plt.show()