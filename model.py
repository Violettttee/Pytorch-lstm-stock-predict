# 定义模型
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
# 定义一个类
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
        super(Net,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM

        self.liner1 = nn.Linear(1, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        self.liner2 = nn.Linear(hidden_size, output_size)

        self.lstm=nn.LSTM(input_size=128,hidden_size=hidden_size,batch_first=True) # LSTM层
        self.lstm2=nn.LSTM(input_size=hidden_size,hidden_size=64,batch_first=True)
    def forward(self,x):
        x = self.liner1(x)
        # x = self.relu(x)
        x = self.tanh(x)

        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to('cuda')

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # output(5, 30, 64)
        output = output[:,-1,:]
        output = self.liner2(output)
        return output.squeeze()

class SelfAttention(nn.Module):
    def __init__(self, attention_units):
        super(SelfAttention, self).__init__()
        self.attention_units = attention_units
        self.W_query = nn.Linear(attention_units, attention_units)
        self.W_key = nn.Linear(attention_units, attention_units)
        self.W_value = nn.Linear(attention_units, attention_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)

        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.softmax(attention_weights)
        attention_output = torch.matmul(attention_weights, value)

        return attention_output