#!/usr/bin/env python3

import os
import datetime
import numpy as np
import os.path as osp
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib
matplotlib.use('tkagg')
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


def select_action(model, state):
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
    return model, action.item()


def finish_episode(model):
    """Calculates actor and critic loss and performs backprop."""
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + GAMMA * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)
        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()
    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

    return model


def run_episode_ac_learn_ur5etosser(episode):
    """Runs an episode."""
    global model, episode_totalrewards
    observation = utils.gen_random_observation(
        grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
        grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
        grasp_orient_limit=GRASP_ORIENT_P,
        grasp_width_limit=GRASP_WIDTH_P)
    state = observation
    episode_reward = 0

    for t in range(STEPS):  # loop for one episode
        model, action = select_action(model, state)
        open_len_fin, target_vel = utils.gen_toss_params(
            action,
            OPEN_LEN_FIN_PMAX,
            TARGET_VEL_PMAX,
            OPEN_LEN_FIN_MIN,
            TARGET_VEL_MIN)

        robot.restart_sim(observation)  # reset environment
        throw_results = robot.picktoss(
            open_len_fin,
            np.deg2rad(args.transport_vel),
            target_vel,
            is_print_state=False)
        reward = utils.get_reward(throw_results, reward_func_type)
        model.rewards.append(reward)
        episode_reward += reward

        observation = utils.gen_random_observation(
            grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
            grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
            grasp_orient_limit=GRASP_ORIENT_P,
            grasp_width_limit=GRASP_WIDTH_P)
        state = observation
    # perform backprop
    model = finish_episode(model)

    # processes at the end
    episode_totalrewards.append(episode_reward)
    if IS_PRINT_STATE:
        if episode % LOG_INTERVAL == 0:
            print('-> %d episode after %f time steps: mean reward %f' %
                  (episode, t + 1, np.mean(episode_totalrewards)))
    if IS_PLOT_PROGRESS:
        utils.plot_rewards_transition(
            episode_totalrewards, avg_interval=AVERAGE_INTERVAL)

    if episode % SAVE_WEIGHT_INTERVAL == 0:
        now = datetime.datetime.now()
        datestr = "{0:%Y%m%d_%H%M%S}".format(now)
        if IS_SAVE_WEIGHT:
            filename = datestr+"_episode_"+str(episode)+".pth"
            savewgtfilepath = osp.join(savewgtdirpath, filename)
            torch.save(model.state_dict(), savewgtfilepath)
        if IS_SAVE_PROGRESS:
            filename = datestr+"_episode_"+str(episode)+".npy"
            savelogfilepath = osp.join(savelogdirpath, filename)
            np.save(savelogfilepath,
                    np.array(episode_totalrewards, dtype=np.float64))

    return action


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

    reward_func_type = args.reward_func

    # env.seed(SEEDS)
    # torch.manual_seed(SEEDS)  # to fix random values
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    model = Network()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    eps = np.finfo(np.float32).eps.item()

    if args.weightpath not in [None, 'None']:
        print("\nLoading weight file named "+args.weightpath+" ...")
        model.load_state_dict(torch.load(args.weightpath))

    episode_totalrewards = []
    final_actions = []
    if IS_SAVE_WEIGHT:
        savewgtdirpath = osp.join(
            osp.dirname(__file__), "../weight",
            args.obj, "ac", args.reward_func)
        os.makedirs(savewgtdirpath, exist_ok=True)
    if IS_SAVE_PROGRESS:
        savelogdirpath = osp.join(
            osp.dirname(__file__), "../log",
            args.obj, "ac", args.reward_func)
        os.makedirs(savelogdirpath, exist_ok=True)
    for e in range(EPISODES):
        final_action = run_episode_ac_learn_ur5etosser(e)
        final_actions.append(final_action)

    if IS_PRINT_STATE:
        open_len_fin, target_vel = utils.gen_toss_params(
            final_actions[-1],
            OPEN_LEN_FIN_PMAX,
            TARGET_VEL_PMAX,
            OPEN_LEN_FIN_MIN,
            TARGET_VEL_MIN)

        print("****************")
        print("Final actions after training.")
        print(" -> Hand opening length: "+str(open_len_fin)+" [m]")
        print(" -> Release velocity: "+str(target_vel)+" [m/s]")
