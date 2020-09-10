import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(0)

data = pd.read_csv('삼성전자_label.csv', index_col=0)
data = data[['open', 'high', 'low', 'volume', 'adj_close']]
train = np.array(data['2014-01-01':'2017-12-31'])
test = np.array(data['2018-01-01':'2018-12-31'])

def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []

    for i in range(len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]

        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)


train = minmax_scaler(train)
test = minmax_scaler(test)

trainX, trainY = build_dataset(train, seq_length)
testX, testY = build_dataset(test, seq_length)

trainX_tensor = torch.FloatTensor(trainX).cuda()
trainY_tensor = torch.FloatTensor(trainY).cuda()

testX_tensor = torch.FloatTensor(testX).cuda()
testY_tensor = torch.FloatTensor(testY).cuda()

seq_length = 7
data_dim = 5
hidden_dim = 2
output_dim = 1
learning_rate = 0.01
iterations = 10

"""
LSTM

"""

class LSTM(torch.nn.Module):
    """
    class
    __init__
    forward
    __call__ : DataLoader를 직접 작성할 때 옐드를 활용하면 됨
    __getitem__
    """

    def __init__(self,input_dim,hidden_dim,output_dim,layers):
        super(LSTM, self).__init()
        self.lstm = torch.nn.LSTM(input_dim, hiffen_dim, numy_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim,bias=True)
        # self.h0

    def forward(self, x):
        output, hidden = self.lstm(x)
        output, (hidden, call) = self.lstm(x,(h0,c0))
        print('output: ',output)
        print('with: ',output[:,-1])
        print('hidden: ',hidden)
        #output = self.fc(hidden)

net = LSTM(data_dim, hidden_dim, output_dim, 1)
net.cuda()

criterion = torch.nn.MSELoos()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for i in range(iterations):
    optimizer.zero_grad()
    outputs = net(trainX_tensor)

    loss = criterion(outputs, trainY_tensor)
    loss.backward()
    optimizer.step()

plt.plot(testY_tensor.cpu())
plt.plot(net(testX_tensor).cpu().detach().numpy())
plt.legend(['original','prediction'])
plt.show()
