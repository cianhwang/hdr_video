from convolutional_rnn import Conv2dLSTM
import torch
'''
rnn = Conv2dLSTM(8, 16, 7, stride = 1)
inputs = torch.randn(3, 5, 8, 227, 227)
outputs, (h, c) = rnn(inputs, None)
print("inputs.size()", inputs.size())
print("outputs.size()", outputs.size())
print("h.size()", h.size())
print("c.size()", c.size())
'''

in_channels = 8
net = Conv2dLSTM(in_channels=in_channels,  # Corresponds to input size
                                   out_channels=8,  # Corresponds to hidden size
                                   kernel_size=5,  # Int or List[int]
                                   num_layers=1,
                                   bidirectional=False,bias=False,
                                   dilation=1, stride=1)
length = 1
batchsize = 4
shape = (1024, 1024)
x = torch.randn(length, batchsize, in_channels, *shape)
y, (h, c) = net(x)
print("x.size()", x.size())
print("y.size()", y.size())
print("h.size()", h.size())
print("c.size()", c.size())

print(torch.sum((y-h)**2))



