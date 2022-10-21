#!/usr/bin/env python3

import time
import numpy as np
from config.simparams import PT_STATE_EXAMPLES, IS_PRINT_STATE

import utils
import cameras
from robot import Robot


def main(args):
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

    mpph = 0
    mpph_sum = 0
    elapsed_time_sum = 0
    elapsed_time_sum_hours = 0
    while True:
        if args.measure_time:
            start = time.time()
        robot.picktoss(
            args.release_width,
            np.deg2rad(args.transport_vel),
            args.throw_vel,
            is_print_state=IS_PRINT_STATE)
        if args.measure_time:
            elapsed_time = time.time() - start
            print("* elapsed_time: {0}"
                  .format(elapsed_time) + " [sec]")
            elapsed_time_sum += elapsed_time
            mpph += 1
            if elapsed_time_sum > 7200:  # 2 hours
                mpph_sum += mpph
                elapsed_time_sum_hours += elapsed_time_sum / 3600.0
                mpph_result = mpph_sum / elapsed_time_sum_hours
                print("*** mean picks per hour: {0}"
                      .format(mpph_result) + " [picks]")
                elapsed_time_sum = 0
                mpph = 0
        robot.restart_sim(setStates, isMovingObj=args.moving_obj)


if __name__ == '__main__':
    main(utils.get_option())
