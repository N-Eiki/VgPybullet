#!/usr/bin/env python3

import os
import copy
import random
import datetime
import numpy as np
import os.path as osp
from sympy import E
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib
matplotlib.use('tkagg')
import utils

import cameras
from robot import Robot
from config.simparams import *


class ReplayMemory:
    """Memory to replay actions in the training."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


def select_action(Q, state, episode):
    """Selects an action based on the state."""
    epsilon = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * episode / EPS_DECAY)
    if epsilon <= np.random.uniform(0, 1):
        with torch.no_grad():
            state_tensor = state.type(FloatTensor)
        return Q(state_tensor).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[np.random.choice(list(range(ACTIONS)))]])


def run_episode_ddqn_learn_ur5etosser(episode):
    """Runs an episode."""
    global Q, tQ, memory, episode_totalrewards
    observation = utils.gen_random_observation(
        grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
        grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
        grasp_orient_limit=GRASP_ORIENT_P,
        grasp_width_limit=GRASP_WIDTH_P)
    state = observation
    episode_reward = 0
    tQ = copy.deepcopy(Q)  # copy weights

    for t in range(STEPS):  # loop for one episode
        action = select_action(Q, FloatTensor([state]), episode)
        open_len_fin, target_vel = utils.gen_toss_params(
            action[0, 0].item(),
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
        reward = utils.get_reward(throw_results, reward_func_type)
        episode_reward += reward  # stact the reward

        observation = utils.gen_random_observation(
            grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
            grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
            grasp_orient_limit=GRASP_ORIENT_P,
            grasp_width_limit=GRASP_WIDTH_P)
        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([observation]),
                     FloatTensor([reward])))
        learn(Q, tQ, memory)
        state = observation
        tQ = copy.deepcopy(Q)  # copy weights

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
            torch.save(Q.state_dict(), savewgtfilepath)
        if IS_SAVE_PROGRESS:
            filename = datestr+"_episode_"+str(episode)+".npy"
            savelogfilepath = osp.join(savelogdirpath, filename)
            np.save(
                savelogfilepath,
                np.array(episode_totalrewards, dtype=np.float64))

    return action


def learn(Q, tQ, memory):
    """Optimizes model weights."""
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = \
        zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = Q(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = tQ(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(torch.squeeze(current_q_values), expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
    Q = Network()
    if use_cuda:
        Q.cuda()
    memory = ReplayMemory(10000)
    optimizer = optim.Adam(Q.parameters(), LR)

    tQ = Network()  # same network structure
    if use_cuda:
        tQ.cuda()

    if args.weightpath not in [None, 'None']:
        print("\nLoading weight file named "+args.weightpath+" ...")
        Q.load_state_dict(torch.load(args.weightpath))
        tQ.load_state_dict(torch.load(args.weightpath))

    episode_totalrewards = []
    final_actions = []
    if IS_SAVE_WEIGHT:
        savewgtdirpath = osp.join(
            osp.dirname(__file__), "../weight",
            args.obj, "ddqn", args.reward_func)
        os.makedirs(savewgtdirpath, exist_ok=True)
    if IS_SAVE_PROGRESS:
        savelogdirpath = osp.join(
            osp.dirname(__file__), "../log",
            args.obj, "ddqn", args.reward_func)
        os.makedirs(savelogdirpath, exist_ok=True)
    for e in range(EPISODES):
        final_action = run_episode_ddqn_learn_ur5etosser(e)
        final_actions.append(final_action)

    if IS_PRINT_STATE:
        open_len_fin, target_vel = utils.gen_toss_params(
            final_actions[-1][0, 0].item(),
            OPEN_LEN_FIN_PMAX,
            TARGET_VEL_PMAX,
            OPEN_LEN_FIN_MIN,
            TARGET_VEL_MIN)

        print("****************")
        print("Final actions after training.")
        print(" -> Hand opening length: "+str(open_len_fin)+" [m]")
        print(" -> Release velocity: "+str(target_vel)+" [m/s]")
