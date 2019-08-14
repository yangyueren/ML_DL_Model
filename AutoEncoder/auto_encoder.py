import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()


optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


view_data = train_data.data[:N_TEST_IMG,:,:].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    plt.figure(i)
    plt.imshow(np.reshape(view_data.data.numpy()[i], (28,28)), cmap='gray')
    plt.show()

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        """
        b_x 不需要除255
        mnist数据集的dataset每次迭代返回的是一个转为灰度图的PIL的Image, 
        然后灰度图再转为tensor的时候，就变为0~1了。
        """
        b_x = x.view(-1,28*28)
        b_y = x.view(-1, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Epoch: {epoch} | train loss: {loss.data.numpy():.4f}')


_, x = autoencoder(view_data)
for i in range(N_TEST_IMG):
    plt.figure(i)
    plt.imshow(np.reshape(x.data.numpy()[i], (28,28)), cmap='gray')
    plt.show()