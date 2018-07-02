import torch as t
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
t.manual_seed(1)
x=t.unsqueeze(t.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*t.rand(x.size())
x,y=Variable(x),Variable(y)
plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()
class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_outdden):
        super(Net,self).__init__()
        self.hidden=t.nn.Linear(n_feature,n_hidden)
        self.relu=t.nn.ReLU()
        self.predict=t.nn.Linear(n_hidden,n_outdden)
    def forward(self,x):
        x=self.hidden(x)
        x=self.relu(x)
        x=self.predict(x)
        return x
net=Net(n_feature=1,n_hidden=10,n_outdden=1)
print(net)
loss_func=t.nn.MSELoss()
optimizer=t.optim.SGD(net.parameters(),lr=0.1)
plt.ion()
for i in tqdm(range(1000)):
    prediction=net(x)
    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r--',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.item(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()