import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from utils import *
from model import Net

input_size = 1
hidden_size = 256
num_layers = 1
output_size = 1
batch_size = 32
seq_length = 3
lr = 2*1e-6
path = './datasets/all_stocks_2006-01-01_to_2018-01-01.csv'

model_list = []
optimizer_list = []

tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

df = pd.read_csv(path)
AAPL = df.loc[df['Name'] == tech_list[0]]
GOOGL = df.loc[df['Name'] == tech_list[1]]
MSFT = df.loc[df['Name'] == tech_list[2]]
AMZN = df.loc[df['Name'] == tech_list[3]]
company_list = [AAPL, GOOGL, MSFT, AMZN]

for i in range(len(tech_list)):
    temp_model = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr)
    temp_model.load_state_dict(torch.load(f"state_dict_of_{tech_list[i]}"))
    model_list.append(temp_model)
    optimizer_list.append(temp_optimizer)

data = [company.filter(['Close'])  for company in company_list]   # (4,seq_length) type:DataFrame
dataset = [data[i].values for i in range(len(data))]  # (4,seq_length) type:List

scaler = [MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(-1, 1)),
              MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(-1, 1))]

