


import torch


bs = 32
imgs = torch.ones(bs, 100)
att = torch.zeros(bs, 20)

z = torch.cat((imgs, att), dim=1)

print(z)


def func1(a, b):
    return a + b


bb = func1
print(bb(1, 2))