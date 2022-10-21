#!/usr/bin/env python3

import os
import os.path as osp
import torch
from torchviz import make_dot

from config.simparams import *
from train_dqn_ur5etosser import Network as DQN
from train_ddqn_ur5etosser import Network as DDQN
from train_ac_ur5etosser import Network as AC


def save_network_graph(model, dpath, fname, imgext="png"):
    with torch.no_grad():
        state_tensor = torch.randn(1, STATES).type(FloatTensor)  # dummy input
    q_out = model(state_tensor)
    image = make_dot(q_out, params=dict(model.named_parameters()))
    image.format = imgext
    savefilepath = osp.join(dpath, fname)
    image.render(savefilepath)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    Tensor = FloatTensor

    dqn = DQN()
    if use_cuda:
        dqn.cuda()
    ddqn = DDQN()
    if use_cuda:
        ddqn.cuda()
    ac = AC()
    if use_cuda:
        ac.cuda()

    savedirpath = osp.join(
        osp.dirname(__file__), "../output/network")
    os.makedirs(savedirpath, exist_ok=True)
    save_network_graph(dqn, savedirpath, "dqn")
    save_network_graph(ddqn, savedirpath, "ddqn")
    save_network_graph(ac, savedirpath, "ac")
