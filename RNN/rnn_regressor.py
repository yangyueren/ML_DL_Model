"""
:reference
https://morvanzhou.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/
http://jermmy.xyz/2018/11/25/2018-11-25-how-to-write-rnn-in-pytorch-and-tensorflow/
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

plt.figure(0)
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()


"""
class torch.nn.RNN(*args, **kwargs)

Parameters: input_size, hidden_size, num_layers, …

Inputs: input, h_0

input of shape (seq_len, batch, input_size)
h_0 of shape (num_layers * num_directions, batch, hidden_size)
Outputs: output, h_n

output of shape (seq_len, batch, num_directions * hidden_size)
h_n (num_layers * num_directions, batch, hidden_size)

每次 forward 后计算得到的 hidden state。
毕竟 h_n 只保留了最后一步的 hidden state，但中间的 hidden state 也有可能会参与计算，
所以 pytorch 把中间每一步输出的 hidden state 都放到 output 中，
因此，你可以发现这个 output 的维度是 (seq_len, batch, num_directions * hidden_size)。
"""
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)

    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b--')
    plt.draw();plt.pause(0.05)

plt.ioff()
plt.show()
