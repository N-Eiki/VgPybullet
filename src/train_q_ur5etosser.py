#!/usr/bin/env python3

import os
import datetime
import numpy as np
import os.path as osp

import matplotlib
matplotlib.use('tkagg')
import utils

import cameras
from robot import Robot
from config.simparams import *


def bins(clip_min, clip_max, num):
    """Generate bins."""
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):
    """Converts observed states to degitized values."""
    grasp_p_height, grasp_p_slide, grasp_orient, grasp_width = observation
    digitized = [
        np.digitize(grasp_p_height, bins=bins(
            GRASP_POSI_HEIGHT_P[0], GRASP_POSI_HEIGHT_P[1], DIVISIONS)),
        np.digitize(grasp_p_slide, bins=bins(
            GRASP_POSI_SLIDE_P[0], GRASP_POSI_SLIDE_P[1], DIVISIONS)),
        np.digitize(grasp_orient, bins=bins(
            GRASP_ORIENT_P[0], GRASP_ORIENT_P[1], DIVISIONS)),
        np.digitize(grasp_width, bins=bins(
            GRASP_WIDTH_P[0], GRASP_WIDTH_P[1], DIVISIONS)),
    ]
    return sum([x * (DIVISIONS**i) for i, x in enumerate(digitized)])


def select_action(q_table, next_state, episode):
    """Selects next action with e-greedy"""
    epsilon = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * episode / EPS_DECAY)
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice(list(range(ACTIONS)))
    return next_action


def update_Qtable(q_table, state, action, reward, next_state):
    """Updates Q table."""
    next_Max_Q = max(q_table[next_state][0], q_table[next_state][1])
    q_table[state, action] = \
        (1 - ALPHA) * q_table[state, action] + \
        ALPHA * (reward + GAMMA * next_Max_Q)
    return q_table


def run_episode_q_learn(episode):
    """Runs an episode."""
    global q_table, episode_totalrewards
    observation = utils.gen_random_observation(
        grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
        grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
        grasp_orient_limit=GRASP_ORIENT_P,
        grasp_width_limit=GRASP_WIDTH_P)
    state = digitize_state(observation)
    episode_reward = 0

    for t in range(STEPS):  # loop for one episode
        # calculate action a_{t}
        action = select_action(q_table, state, episode)

        # calculating s_{t+1}, r_{t} by executing a_t
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
        episode_reward += reward  # stact the reward

        # update q table by calculating digitized status s_{t+1}
        observation = utils.gen_random_observation(
            grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
            grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
            grasp_orient_limit=GRASP_ORIENT_P,
            grasp_width_limit=GRASP_WIDTH_P)

        next_state = digitize_state(observation)  # digitize status at t+1
        q_table = update_Qtable(q_table, state, action, reward, next_state)
        state = next_state

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
            filename = datestr+"_episode_"+str(episode)+".npy"
            savewgtfilepath = osp.join(savewgtdirpath, filename)
            np.save(savewgtfilepath, q_table)
        if IS_SAVE_PROGRESS:
            filename = datestr+"_episode_"+str(episode)+".npy"
            savelogfilepath = osp.join(savelogdirpath, filename)
            np.save(
                savelogfilepath,
                np.array(episode_totalrewards, dtype=np.float64))

    return action


if __name__ == '__main__':
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

    q_table = np.random.uniform(
        low=-1, high=1, size=(DIVISIONS**STATES, ACTIONS))
    if args.weightpath not in [None, 'None']:
        print("\nLoading weight file named "+args.weightpath+" ...")
        q_table = np.load(args.weightpath)

    episode_totalrewards = []
    final_actions = []
    if IS_SAVE_WEIGHT:
        savewgtdirpath = osp.join(
            osp.dirname(__file__), "../weight",
            args.obj, "q", args.reward_func)
        os.makedirs(savewgtdirpath, exist_ok=True)
    if IS_SAVE_PROGRESS:
        savelogdirpath = osp.join(
            osp.dirname(__file__), "../log",
            args.obj, "q", args.reward_func)
        os.makedirs(savelogdirpath, exist_ok=True)
    for e in range(EPISODES):
        final_action = run_episode_q_learn(e)
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
