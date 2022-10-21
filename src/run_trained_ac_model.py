#!/usr/bin/env python3

import os
import numpy as np
import os.path as osp
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import utils
import cameras
from robot import Robot
from config.simparams import *


class Network(nn.Module):
    """implements both actor and critic in one model."""
    def __init__(self):
        super(Network, self).__init__()
        self.affine1 = nn.Linear(STATES, HIDDEN_SIZE)
        # actor's layer
        self.action_head = nn.Linear(HIDDEN_SIZE, ACTIONS)
        # critic's layer
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """forward of both actor and critic."""
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)
        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)
        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def select_action(state):
    """Selects an action based on the state."""
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution
    # over the list of probabilities of actions
    m = Categorical(probs)
    # and sample an action using the distribution
    action = m.sample()
    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def run_one_motion_ac_learn_ur5etosser():
    """Runs one motion."""
    state = utils.gen_random_observation(
        grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
        grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
        grasp_orient_limit=GRASP_ORIENT_P,
        grasp_width_limit=GRASP_WIDTH_P)

    action = select_action(state)
    open_len_fin, target_vel = utils.gen_toss_params(
        action,
        OPEN_LEN_FIN_PMAX,
        TARGET_VEL_PMAX,
        OPEN_LEN_FIN_MIN,
        TARGET_VEL_MIN)

    robot.restart_sim(state)  # reset environment
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

    # env.seed(SEEDS)
    # torch.manual_seed(SEEDS)  # to fix random values
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    model = Network()
    if args.weightpath in [None, 'None']:
        print("Error: Please specify the weight file path with --weightpath.")

    print("\nLoading weight file named "+args.weightpath+" ...")
    model.load_state_dict(torch.load(args.weightpath))
    run_one_motion_ac_learn_ur5etosser()
