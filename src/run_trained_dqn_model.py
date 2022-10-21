#!/usr/bin/env python3

import os
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import cameras
from robot import Robot
from config.simparams import *


class Network(nn.Module):
    """Deep Q network."""
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(STATES, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, ACTIONS)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = F.relu(self.fc4(h))
        return y


def select_action(state):
    """Selects an action based on the state."""
    with torch.no_grad():
        state_tensor = state.type(FloatTensor)
    return Q(state_tensor).data.max(1)[1].view(1, 1)


def run_one_motion_dqn_learn_ur5etosser():
    """Runs one motion."""
    state = utils.gen_random_observation(
        grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
        grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
        grasp_orient_limit=GRASP_ORIENT_P,
        grasp_width_limit=GRASP_WIDTH_P)

    action = select_action(FloatTensor([state]))
    open_len_fin, target_vel = utils.gen_toss_params(
        action[0, 0].item(),
        OPEN_LEN_FIN_PMAX,
        TARGET_VEL_PMAX,
        OPEN_LEN_FIN_MIN,
        TARGET_VEL_MIN)

    robot.restart_sim(state)
    throw_results = robot.picktoss(
        open_len_fin,
        np.deg2rad(args.transport_vel),
        target_vel,
        is_print_state=False)


if __name__ == '__main__':
    # if gpu is to be used
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    Tensor = FloatTensor

    args = utils.get_option()
    if args.cam_cfg == 'rs-d415':
        cam_cfgs = cameras.RealSenseD415.CONFIG
    cam_cfg = cam_cfgs[3]
    robot = Robot(args.renders,
                  args.sim_view,
                  np.float32([[], [], []]),
                  args.urdf_dir,
                  args.gripper_type,
                  args.obj,
                  args.frame,
                  cam_cfg,
                  args.box_dist,
                  args.save_snapshots)

    Q = Network()
    if use_cuda:
        Q.cuda()

    if args.weightpath in [None, 'None']:
        print("Error: Please specify the weight file path with --weightpath.")

    print("\nLoading weight file named "+args.weightpath+" ...")
    Q.load_state_dict(torch.load(args.weightpath))
    run_one_motion_dqn_learn_ur5etosser()
