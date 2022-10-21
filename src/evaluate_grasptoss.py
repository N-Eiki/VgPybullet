#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils
import cameras
from robot import Robot
from config.simparams import PT_STATE_EXAMPLES, IS_PRINT_STATE

matplotlib.use('tkagg')


def autolabel(rects, ax):
    """Labels numbers for bars."""
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate(
            '{}'.format(int(height)),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')


def autolabels(rect1, rect2, ax):
    """Labels numbers for bars."""
    for r1, r2 in zip(rect1, rect2):
        height1 = r1.get_height()
        height2 = r1.get_height() + r2.get_height()
        if height1 == 0 or height2 == 0:
            continue
        ax.annotate(
            '{}'.format(int(height1)),
            xy=(r1.get_x() + r1.get_width() / 2, height1),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')
        ax.annotate(
            '{}'.format(int(height2)),
            xy=(r2.get_x() + r2.get_width() / 2, height2),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')


def plot_statistics(rs, rf, ls, lf, cls, clf):
    """Plots tossing results."""
    # plot releasing results
    labels = ['Release success',
              'Release failure']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    p1 = ax.bar(x, [rs, rf], width)
    ax.set_ylabel('Number of trials')
    ax.set_xticks(x, labels)
    ax.set_yticks(np.arange(0, STEPS+int(STEPS*0.2), 10))
    autolabel(p1, ax)

    # plot landing results
    labels = ['Land success',
              'Land failure']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    p1 = ax.bar(x, [cls, clf], width)
    p2 = ax.bar(x, [ls-cls, lf-clf], width, bottom=[cls, clf])
    ax.set_ylabel('Number of trials')
    ax.set_xticks(x, labels)
    ax.set_yticks(np.arange(0, STEPS+int(STEPS*0.2), 10))
    ax.legend((p1[0], p2[0]), ('Contacted', 'No contacted'))
    autolabels(p1, p2, ax)

    fig, ax = plt.subplots()
    edges = range(0, 460, 10)
    n, bins, patches = ax.hist(rotation_degree, color='red', bins=edges)
    ax.set_xlabel('Rotation angle [deg]')
    ax.set_ylabel('Frequency')
    ax.set_yticks(np.arange(0, STEPS+int(STEPS*0.2), 10))
    for i in range(17, 26, 1):
        patches[i].set_facecolor('indianred')
    for i in range(26, 35, 1):
        patches[i].set_facecolor('lightcoral')
    for i in range(35, 45, 1):
        patches[i].set_facecolor('rosybrown')
    autolabel(patches, ax)
    plt.show()


def evaluate_tossing():
    """Simulates and evaluates tossing results."""
    for step in range(STEPS):
        throw_results = \
            robot.picktoss(
                args.release_width,
                np.deg2rad(args.transport_vel),
                args.throw_vel,
                is_print_state=IS_PRINT_STATE)
        robot.restart_sim(setStates, isMovingObj=args.moving_obj)

        release_success.append(throw_results['release_success'])
        land_success.append(throw_results['land_success'])
        wall_contact_land.append(False)
        wall_contact_noland.append(False)
        if throw_results['type'] == 'wall_contact':
            if throw_results['land_success']:
                wall_contact_land[-1] = True
            else:
                wall_contact_noland[-1] = True
        rotation_degree.append(throw_results['rot_deg'])

    # plot statistics of tossing results
    rs_sum = np.sum(release_success)
    rf_sum = STEPS - np.sum(release_success)
    ls_sum = np.sum(land_success)
    lf_sum = STEPS - np.sum(land_success)
    cls_sum = np.sum(wall_contact_land)
    clf_sum = np.sum(wall_contact_noland)
    plot_statistics(rs_sum, rf_sum, ls_sum, lf_sum, cls_sum, clf_sum)


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
    setStates = PT_STATE_EXAMPLES[args.obj]
    robot.restart_sim(setStates, isMovingObj=args.moving_obj)

    # get tossing results
    release_success = []
    land_success = []
    wall_contact_land = []
    wall_contact_noland = []
    rotation_degree = []
    STEPS = args.num_step

    evaluate_tossing()
