#!/usr/bin/env python3

import time
import numpy as np
from config.simparams import PP_STATE_EXAMPLES, IS_PRINT_STATE

import utils
import cameras
from robot import Robot


def main(args):
    # --------------- Setup options ---------------
    num_obj = args.num_obj
    training_target = args.training_target
    no_regulate_residual = args.no_regulate_residual
    input_frames = args.input_frames
    

    camera_diff = 0.5
    if training_target=="static_grasp":
        camera_diff = 0

    workspace_limits = np.array([
        [0.5-0.224, 0.5+0.224], [-0.224 - camera_diff, 0.224 - camera_diff], [-0.0001, 0.4]
    ])

    heightmap_resolution = args.heightmap_resolution
    random_seed = args.random_seed
    force_cpu = args.force_cpu
    preload = args.preload
    # ------------- Algorithm options -------------

    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only

     # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    load_static_snapshot = args.load_static_snapshot
    static_snapshot_file = os.path.abspath(args.static_snapshot_file) if load_static_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
    
    if args.cam_cfg == 'rs-d415':
        cam_cfgs = cameras.RealSenseD415.CONFIG
    cam_cfg = cam_cfgs[3]
    robot = Robot(args.renders,
                  args.sim_view,
                  args.urdf_dir,
                  args.gripper_type,
                  args.obj,
                  args.frame,
                  cam_cfg,
                  args.box_dist,
                  args.save_snapshots,
                  num_obj,
                  workspace_limits,
                  is_testing,
                  test_preset_cases, 
                  test_preset_file,
                  training_target
                  )

    setStates = PP_STATE_EXAMPLES[args.obj]
    robot.restart_sim(setStates, isMovingObj=args.moving_obj)

    mpph = 0
    mpph_sum = 0
    elapsed_time_sum = 0
    elapsed_time_sum_hours = 0
    while True:
        robot.get_camera_data()
        if args.measure_time:
            start = time.time()
        workspace_limits = np.float32([[0.5, 1.0], [0., 0.5], [0.3, 1.50]])
        x, y = 224, 224
        primitive_position = [workspace_limits[0][0] + (x-112)*0.002, (y-112)*0.002+workspace_limits[1][0], workspace_limits[2][0]]
        # primitive_position =[0.5, .0, 0.1]
        best_rotation_angle = np.deg2rad(0*(360.0/16))
        robot.grasp(primitive_position, best_rotation_angle, workspace_limits, np.deg2rad(args.transport_vel))
        # robot.pickplace(
        #     np.deg2rad(args.transport_vel),
        #     is_print_state=IS_PRINT_STATE)
        if args.measure_time:
            elapsed_time = time.time() - start
            print("* elapsed_time: {0}"
                  .format(elapsed_time) + "[sec]")
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
