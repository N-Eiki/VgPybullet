#!/usr/bin/env python3

import os
import numpy as np
import os.path as osp

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


def run_one_motion_q_learn():
    """Runs one motion."""
    observation = utils.gen_random_observation(
        grasp_posi_height_limit=GRASP_POSI_HEIGHT_P,
        grasp_posi_slide_limit=GRASP_POSI_SLIDE_P,
        grasp_orient_limit=GRASP_ORIENT_P,
        grasp_width_limit=GRASP_WIDTH_P)
    state = digitize_state(observation)
    action = np.argmax(q_table[state])

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

    if args.weightpath in [None, 'None']:
        print("Error: Please specify the weight file path with --weightpath.")

    print("\nLoading weight file named "+args.weightpath+" ...")
    q_table = np.load(args.weightpath)
    run_one_motion_q_learn()
