#!/usr/bin/env python3

import numpy as np
import os.path as osp
import tkinter
import tkinter.filedialog
import tkinter.messagebox

import utils


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "*")]
    iDir = osp.join(osp.abspath(
        osp.dirname(__file__)), "../log")
    tkinter.messagebox.showinfo(
        'please choose one of npy files.')
    file = tkinter.filedialog.askopenfilename(
        filetypes=fTyp, initialdir=iDir)

    episode_totalrewards = np.load(file)
    utils.plot_rewards_transition(
        episode_totalrewards, pausesec=30)
