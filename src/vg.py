#!/usr/bin/env python3

import os
import sys
import cv2
import time
import torch
import random
import threading
import numpy as np
import copy
import pybullet as p

from trainer import Trainer
from logger import Logger

from config.simparams import PP_STATE_EXAMPLES, IS_PRINT_STATE

import utils
import cameras
from robot import Robot


def main(args):
    # --------------- Setup options ---------------
    method = "reinforcement"
    num_obj = args.num_obj
    training_target = args.training_target
    no_regulate_residual = args.no_regulate_residual
    input_frames = args.input_frames
    is_sim = True
    transport_vel = np.deg2rad(args.transport_vel)
    setStates = PP_STATE_EXAMPLES[args.obj]
    debug = args.debug
    camera_diff = 0.5
    if training_target=="static_grasp":
        camera_diff = 0

    workspace_limits = np.array([
        [0.5-0.224, 0.5+0.224], [-0.224, 0.224], [-0.0001, 0.24]
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
    teacher_student = args.teacher_student
    multi_steps = False
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
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
    
    save_base = None
    if load_snapshot:
        save_base = "/".join(snapshot_file.split("/")[:-2])

    # Set random seed
    np.random.seed(random_seed)


    

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
                  training_target,
                  debug
                  )

    trainer = Trainer(robot, training_target, future_reward_discount, input_frames,
                      is_testing, load_snapshot, snapshot_file, force_cpu, load_static_snapshot, static_snapshot_file)
    

    # Initialize data logger
    exp_keyword = f"{input_frames}Frames-"
    if args.no_regulate_residual:
        exp_keyword += "NoRegulateResidual-"

    if args.teacher_student:
        exp_keyword += "TeacherStudent-" 
    if save_base:
            logger = Logger(True, save_base, training_target, exp_keyword)
    else:
            logger = Logger(False, logging_directory, training_target, exp_keyword)
    # logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters
    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if preload:
        trainer.preload(logger.transitions_directory)
        logger.load_logs()

    explore_prob = 0.5 if not is_testing else 0.0


    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'grasp_success' : False,
                          'grasp_success_count':0,
                          'touch_object' : 0,
                          'no_touch_count':0,
                          'regulate_residual':0,
                          'last_100_times_success':[],
                          'last_10_times_success':[],
                          'multi_pix_ind':None,
                          'regulate_success' : True,
                          "infer_start":0,
                          }

    setStates = PP_STATE_EXAMPLES[args.obj]
    robot.restart_sim(setStates, isMovingObj=args.moving_obj)

    
    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:
                
                explore_actions = False
                explore_actions = np.random.uniform() < explore_prob
                if explore_actions:
                    print(f'strategy: explore {explore_prob}')
                else:
                    print(f'strategy: exploit {explore_prob}')
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)
                # Determine whether grasping or pushing should be executed based on network predictions
                best_grasp_conf = np.max(grasp_predictions)
                print('Primitive vel and confidence scores:  %f (grasp)' % (best_grasp_conf))
                nonlocal_variables['primitive_action'] = 'grasp'
                
                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                use_heuristic = False

                # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                # if nonlocal_variables['primitive_action'] == 'grasp':
                     
                nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                
                predicted_grasp_value = np.max(grasp_predictions)
                if multi_steps:
                    # img_predictions = trainer.visualization_for_find_grasp_point(grasp_predictions)
                    nonlocal_variables['multi_pix_ind'] = trainer.get_multi_targets_pix_ind(grasp_predictions, nonlocal_variables['best_pix_ind'], explore_actions)
                
                if trainer.training_target != 'static_grasp':
                    nonlocal_variables['best_pix_ind_static'] = np.unravel_index(np.argmax(static_grasp_predictions), static_grasp_predictions.shape)
                    predicted_regulate_value = regulate_predictions[nonlocal_variables["best_pix_ind"]].item()
                
                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_grasp_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][1]
                best_pix_y = nonlocal_variables['best_pix_ind'][2]

                if debug:
                    nonlocal_variables['best_pix_ind'] = [0]
                    for _ in range(2):
                        print('------------')
                        data = input("input>>> ")
                        nonlocal_variables['best_pix_ind'].append(int(data))
                    best_pix_x = nonlocal_variables['best_pix_ind'][1]
                    best_pix_y = nonlocal_variables['best_pix_ind'][2]
                    nonlocal_variables['best_pix_ind'] = tuple(nonlocal_variables['best_pix_ind'])
                    data = input("depth >>> ")
                    valid_depth_heightmap[best_pix_y][best_pix_x] = float(data)

                primitive_position = [ (224-best_pix_x) * heightmap_resolution + workspace_limits[0][0],(112-best_pix_y)  * heightmap_resolution, valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
                print('----------------')
                print(primitive_position)

                print(nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],nonlocal_variables['best_pix_ind'][2] )
                print('----------------')
                # if multi_steps:
                #     primitive_positions = []
                #     rotation_angles = []
                #     for point in nonlocal_variables['multi_pix_ind']:
                #         x, y, r = point['x'], point['y'], point['rotate']
                #         primitive_positions.append([(224-x) * heightmap_resolution + workspace_limits[0][0], (224-y) * heightmap_resolution + workspace_limits[1][0]-camera_diff, valid_depth_heightmap[y][x] + workspace_limits[2][0]])
                #         rotation_angles.append(np.deg2rad(r*(360.0/trainer.model.num_rotations)))
                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)
                
                color_heightmap = color_heightmaps[0]
                if teacher_student:
                    color_heightmap = color_heightmaps[-1]
                # Visualize executed primitive, and affordances
                if save_visualizations:
                    best_indices = list(nonlocal_variables['best_pix_ind'])
                    if input_frames==4:
                        best_indices[1]+=75
                    vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, best_indices, 'picture')
                    cv2.imwrite('visualization.picture.png', vis)
                    # cv2.imshow("a", vis)
                    # cv2.waitKey(0)
                    if trainer.training_target != "static_grasp":
                        regulate_pred_vis = trainer.get_prediction_vis(regulate_predictions*20+nonlocal_variables["base_velocity"], color_heightmap, nonlocal_variables['best_pix_ind'], 'conveyor')
                        logger.save_visualizations(trainer.iteration, regulate_pred_vis, 'regulate')
                        cv2.imwrite('visualization.regulate.png', regulate_pred_vis)

                        grasp_pred_vis = trainer.get_prediction_vis(static_grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind_static'])
                        logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                        cv2.imwrite('visualization.static_grasp.png', grasp_pred_vis)

                        vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind_static'], 'picture')
                        cv2.imwrite('visualization.static_picture.png', vis)
                    if multi_steps:
                        grasp_pred_vis = trainer.get_multi_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'], multi_pix_ind=nonlocal_variables['multi_pix_ind'])
                        logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'multi_grasp')
                        cv2.imwrite('visualization.multi_grasp.png', grasp_pred_vis)
                    
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                    np.save('grasp_predictions.npy', grasp_predictions)
                    np.save('regulate_predictions.npy', regulate_predictions)
                # Initialize variables that influence reward
                nonlocal_variables['grasp_success'] = False
                nonlocal_variables['grasp_success'] = 0
                change_detected = False

                def vis_test(trainer,pred,color_heightmap, nonlocal_variables,num):new_pred=np.array([np.zeros((224, 224)) for _ in range(16)]);new_pred[:,num:,:]=pred[:,:224-num,:];idx=list(nonlocal_variables["best_pix_ind"]);idx[1]+=num;grasp_pred_vis = trainer.get_prediction_vis(new_pred, color_heightmap, idx);cv2.imwrite('vistest.png', grasp_pred_vis)
                # def vis_test(trainer,pred,color_heightmap, nonlocal_variables,num):new_pred=np.array([np.zeros((224, 224)) for _ in range(16)]);new_pred[:,:224-num,:]=pred[:,num:,:];idx=list(nonlocal_variables["best_pix_ind"]);idx[1]+=num;grasp_pred_vis = trainer.get_prediction_vis(new_pred, color_heightmap, idx);cv2.imwrite('vistest.png', grasp_pred_vis)
                vis_test(trainer, grasp_predictions, color_heightmap, nonlocal_variables, 75)

             
                if no_regulate_residual:
                    print('option no_regulate_residual is ON, so regulate_residual=0')
                    nonlocal_variables["regulate_residual"] = 0
                
                elif explore_actions :
                    nonlocal_variables["regulate_residual"] = np.random.uniform(-0.01, 0.05)
                    print(f'explore random regulate_residual is : {nonlocal_variables["regulate_residual"]}')
                elif not regulate_predictions is None:#not equal None
                    nonlocal_variables['regulate_residual'] = regulate_predictions[nonlocal_variables["best_pix_ind"]].item()
                    if abs(nonlocal_variables['regulate_residual'])>0.1:
                        nonlocal_variables["regulate_residual"] = np.random.uniform(-0.1, 0.1)
                        print(f'explore random regulate_residual is : {nonlocal_variables["regulate_residual"]}')
                    else:
                        print(f'exploit random regulate_residual is : {nonlocal_variables["regulate_residual"]}')
                
                if nonlocal_variables['primitive_action'] == 'grasp':
                    if trainer.training_target != "static_grasp":
                        nonlocal_variables["regulate_success"] = True
                        nonlocal_variables["regulate_success"] = robot.regulate(nonlocal_variables["regulate_residual"], nonlocal_variables["base_velocity"])
                        if not nonlocal_variables["regulate_success"]:
                            print('\033[31mfailed to regulate conveyor\033[0m')
                            # trainer.iteration-=1

                    if multi_steps:
                        grasp_rate=0
                        for i, (primitive_position, rotation_angle) in enumerate(zip(primitive_positions, rotation_angles)):
                            print(f'doing {i+1}/{len(primitive_positions)} grasp')
                            nonlocal_variables['multi_pix_ind'][i]["grasp_success"],  _ = robot.grasp(primitive_position, best_rotation_angle, workspace_limits, )
                            if nonlocal_variables['multi_pix_ind'][i]["grasp_success"]:
                                nonlocal_variables["grasp_success"] = True
                                
                    else:
                        nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle, workspace_limits, transport_vel)
                        print(f"touch object flg {nonlocal_variables['touch_object']}")
                        print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                # nonlocal_variables['executing_action'] = False
                if nonlocal_variables['regulate_success']:
                    if nonlocal_variables['grasp_success']:
                        nonlocal_variables["grasp_success_count"]+=1
                        logger.success_rate["data"].append(1)
                    else:
                        logger.success_rate["data"].append(0)
                # nonlocal_variables['last_100_times_success'].append(nonlocal_variables['grasp_success'])
                # nonlocal_variables['last_100_times_success'] = nonlocal_variables['last_100_times_success'][-100:]
                # nonlocal_variables['last_10_times_success'].append(nonlocal_variables['grasp_success'])
                # nonlocal_variables['last_10_times_success'] = nonlocal_variables['last_10_times_success'][-10:]*10
                print(f'grasp_success_count : {sum(logger.success_rate["data"])}')
                print(f'success_rate : {sum(logger.success_rate["data"])/ len(logger.success_rate["data"])*100}%')
                print(f'last 100 times success_rate: {sum(logger.success_rate["data"][-100:])} %')
                print(f'last 10 times success_rate: {sum(logger.success_rate["data"][-10:])*10} %')
                if nonlocal_variables["regulate_success"]:
                    if not "all" in logger.success_rate.keys():
                        logger.success_rate["all"] = [sum(logger.success_rate["data"])/ len(logger.success_rate["data"])*100]
                        logger.success_rate["last100"] = [sum(logger.success_rate["data"][-100:])]
                        logger.success_rate["last10"] = [sum(logger.success_rate["data"][-10:])]
                    else:
                        logger.success_rate["all"].append(sum(logger.success_rate["data"])/ len(logger.success_rate["data"])*100)
                        logger.success_rate["last100"].append(sum(logger.success_rate["data"][-100:]))
                        logger.success_rate["last10"].append(sum(logger.success_rate["data"][-10:]))
                nonlocal_variables['executing_action'] = False

            time.sleep(0.1)
            logger.save_success_rate()
            logger.save_loss_transition()

    action_thread = threading.Thread(target=process_actions)
    
    action_thread.daemon = True
    action_thread.start()
    exit_called = False

 
    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Start main training/testing loop
    times = []
    while True:
        print('\n\33[42m%s iteration: %d\33[0m' % ('Testing' if is_testing else 'Training', len(logger.success_rate["data"])))
        iteration_time_0 = time.time()
        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        # time.sleep(1)
        color_heightmaps = []
        valid_depth_heightmaps = []
        robot.env.ur5e.go_initial(transport_vel)
        for frame_count in range(input_frames):
            # print("pause 1 sencond ...")
            # robot.pause_sim()
            # time.sleep(1)
            color_img, depth_img, _ = robot.get_camera_data()
            depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration
            
            depth_img -= 0.75
            # robot.start_sim()
            # print('start from pause ...')

            
            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            # color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
            color_heightmap, depth_heightmap = color_img.copy(), depth_img.copy()
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            color_heightmaps.append(color_heightmap)
            valid_depth_heightmaps.append(valid_depth_heightmap)
            logger.save_images(trainer.iteration, color_img, depth_img, f'0-{frame_count}')
            logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, f'0-{frame_count}')

        # Reset simulation or pause real-world training if table is empty
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
       
            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

        if not exit_called:
            # Run forward pass with network to get affordances
            if trainer.training_target == 'static_grasp':
                regulate_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmaps, valid_depth_heightmaps, is_volatile=True, nonlocal_variables=nonlocal_variables)
            else:
                regulate_predictions, grasp_predictions, nonlocal_variables["base_velocity"], static_grasp_predictions = trainer.forward(color_heightmaps, valid_depth_heightmaps, is_volatile=True, nonlocal_variables=nonlocal_variables, teacher_student=teacher_student)
            # if trainer.iteration>10:
            #     raise ValueError()
            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # # Detect changes
            # depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            # depth_diff[np.isnan(depth_diff)] = 0
            # depth_diff[depth_diff > 0.3] = 0
            # depth_diff[depth_diff < 0.01] = 0
            # depth_diff[depth_diff > 0] = 1
            # change_threshold = 300
            # change_value = np.sum(depth_diff)
            # change_detected = change_value > change_threshold or prev_grasp_success
            # print('Change detected: %r (value: %d)' % (change_detected, change_value))

            
            # Compute training labels
            grasp_label_value, prev_grasp_reward_value = trainer.get_label_value(prev_primitive_action, None, prev_grasp_success, nonlocal_variables["regulate_residual"], prev_grasp_predictions, color_heightmaps, valid_depth_heightmaps,  nonlocal_variables=nonlocal_variables)
            
            regulate_label_value = None
            if trainer.training_target != "static_grasp":
                regulate_label_value, prev_regulate_reward_value = trainer.get_label_value("regulate_grasp", None, prev_grasp_success, nonlocal_variables["regulate_residual"], prev_grasp_predictions, color_heightmaps, valid_depth_heightmaps, nonlocal_variables=nonlocal_variables)
            
            label_value_dict = {
                "grasp":grasp_label_value,
                'regulate':regulate_label_value
            }
            # trainer.label_value_log.append([grasp_label_value])
            trainer.label_value_log.append([label_value_dict])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_grasp_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)
                            
            # Backpropagate
            if not nonlocal_variables["regulate_success"]:
                print("can't regulate conveyor, pass backprop")
            if training_target == "static_grasp":
                if multi_steps:
                    # implement backprop multi step version
                    loss = trainer.multi_steps_backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, prev_primitive_action, prev_best_pix_ind, grasp_label_value, nonlocal_variables=nonlocal_variables)
                else:
                    loss = trainer.backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, prev_primitive_action, prev_best_pix_ind, grasp_label_value, nonlocal_variables=nonlocal_variables)
                if not 'grasp' in logger.loss_logger.keys():
                    logger.loss_logger['grasp'] = [loss]
                else:
                    logger.loss_logger['grasp'].append(loss)
            elif nonlocal_variables["base_velocity"]+nonlocal_variables["regulate_residual"]>0:
                # training_target in ["dynamic_module" , "dynamic_grasp"]
                if training_target != 'dynamic_regulate':
                    if multi_steps:
                        # implement backprop multi step version
                        loss = trainer.multi_steps_backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, prev_primitive_action, prev_best_pix_ind, grasp_label_value, nonlocal_variables=nonlocal_variables)
                    else:
                        loss = trainer.backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, prev_primitive_action, prev_best_pix_ind, grasp_label_value, nonlocal_variables=nonlocal_variables, )
                        if not prev_grasp_success and teacher_student and logger.success_rate["all"][-1]>50:
                            trainer.backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, prev_primitive_action, prev_best_pix_ind, grasp_label_value, nonlocal_variables=nonlocal_variables, teacher_student=teacher_student)
                        
                    if not 'grasp' in logger.loss_logger.keys():
                        logger.loss_logger['grasp'] = [loss]
                    else:
                        logger.loss_logger['grasp'].append(loss)
                if trainer.training_target!='dynamic_grasp':# trainer.training_target == dynamic_regulate or dynamic_module
                    loss = None
                    if prev_grasp_success:

                        loss = trainer.backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, 'regulate_grasp', prev_best_pix_ind, regulate_label_value, regulate_residual=nonlocal_variables["regulate_residual"], nonlocal_variables=nonlocal_variables)
                    elif small_regulate:
                        small_regulate_val = np.random.uniform(-1e-9, 1e-9)
                        loss = trainer.backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, 'regulate_grasp', prev_best_pix_ind, regulate_label_value, regulate_residual=small_regulate_val, nonlocal_variables=nonlocal_variables)
                    
                    if not 'regulate' in logger.loss_logger.keys():
                        logger.loss_logger['regulate'] = [loss]
                    else:
                        logger.loss_logger['regulate'].append(loss)
            else:
                if small_regulate:
                    print("velocity under 0 , so small value backword")
                    small_regulate_val = np.random.uniform(-1e-9, 1e-9)
                    loss = trainer.backprop(prev_color_heightmaps, prev_valid_depth_heightmaps, 'regulate_grasp', prev_best_pix_ind, regulate_label_value, regulate_residual=small_regulate_val, nonlocal_variables=nonlocal_variables)
                else:
                    print("velocity under 0 , so pass backword...")

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.9 * np.power(0.9998, trainer.iteration),0.1) if explore_rate_decay else 0.5

            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    if method == 'reinforcement':
                        sample_reward_value = 0 if prev_grasp_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))
                
                if sample_ind.size > 0:
                    # Find sample with highest surprise value
                    if method == 'reactive':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - (1 - sample_reward_value))
                    elif method == 'reinforcement':
                        label_value_log = [item[0]["grasp"] for item in trainer.label_value_log]
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(label_value_log)[sample_ind[:,0]])
                    
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    # print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmaps = []
                    sample_depth_heightmaps = []
                    for frame_count in range(input_frames):
                        sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, f'%06d.0-{frame_count}.color.png' % (sample_iteration)))
                        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                        sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, f'%06d.0-{frame_count}.depth.png' % (sample_iteration)), -1)
                        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
                        sample_color_heightmaps.append(sample_color_heightmap)
                        sample_depth_heightmaps.append(sample_depth_heightmap)

                    # Compute forward pass with sample
                    with torch.no_grad():
                        if trainer.training_target == 'static_grasp':
                            sample_regulate_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmaps, sample_depth_heightmaps, is_volatile=True, nonlocal_variables=nonlocal_variables)
                        else:
                            sample_regulate_predictions, sample_grasp_predictions, base_velocity, sample_state_feat = trainer.forward(sample_color_heightmaps, sample_depth_heightmaps, is_volatile=True, nonlocal_variables=nonlocal_variables)

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmaps = []
                    next_sample_depth_heightmaps = []
                    for frame_count in range(input_frames):
                        next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, f'%06d.0-{frame_count}.color.png' % (sample_iteration+1)))
                        next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                        next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, f'%06d.0-{frame_count}.depth.png' % (sample_iteration+1)), -1)
                        next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000
                        next_sample_color_heightmaps.append(next_sample_color_heightmap)
                        next_sample_depth_heightmaps.append(next_sample_depth_heightmap)

                    sample_grasp_success = sample_reward_value == 1

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)

                    if training_target == "static_grasp":
                        trainer.backprop(sample_color_heightmaps, sample_depth_heightmaps, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration][0]["grasp"], nonlocal_variables=nonlocal_variables)
                    else:
                        sample_regulate_residual = sample_regulate_predictions[np.unravel_index(np.argmax(sample_grasp_predictions), sample_grasp_predictions.shape)]
                        if training_target != "dynamic_regulate":
                            trainer.backprop(sample_color_heightmaps, sample_depth_heightmaps, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration][0]["grasp"], nonlocal_variables=nonlocal_variables)
                        if training_target != 'dynamic_grasp':
                            trainer.backprop(sample_color_heightmaps, sample_depth_heightmaps, 'regulate_grasp', sample_best_pix_ind, trainer.label_value_log[sample_iteration][0]["regulate"], regulate_residual=sample_regulate_residual, nonlocal_variables=nonlocal_variables)
                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                        # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                    print(f'done experience replay : {trainer.label_value_log[sample_iteration]}')
                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            # Save model snapshot
            if not is_testing:
                model = trainer.model
                if training_target=="static_grasp":
                    model = trainer.static_model
                logger.save_backup_model(model, method)
                if trainer.iteration % 1000 == 0:
                    logger.save_model(trainer.iteration, model, method)
                    if trainer.use_cuda:
                        trainer.static_model = trainer.static_model.cuda()
                        trainer.model = trainer.model.cuda()
                    logger.visualize_loss_transition()
                    logger.visualize_success_rate()

        # Sync both action thread and training thread
        start_loop = time.time()
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)
            # if time.time()-start_loop>10:
            #     break

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmaps = copy.copy(color_heightmaps)
        # prev_depth_heightmaps = copy.copy(depth_heightmaps)
        prev_valid_depth_heightmaps = copy.copy(valid_depth_heightmaps)
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        if regulate_predictions is not None:
            prev_regulate_predictions = regulate_predictions.copy()
        
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']

        random_object_nums = False
        if trainer.iteration%1000==0 and trainer.iteration!=0:
            if training_target=='static_grasp':
                random_object_nums = True
            if 'dynamic' in training_target:
                robot.restart_sim()
                nonlocal_variables["no_touch_count"] = 0
        
        

        if nonlocal_variables["no_touch_count"] > 20:
                robot.restart_sim()
                nonlocal_variables["no_touch_count"] = 0

        elif not nonlocal_variables["grasp_success"]:
            nonlocal_variables["no_touch_count"] +=1
        else:
            nonlocal_variables["no_touch_count"] = 0
        
        if robot.check_all_grasp():
            print('all grasped restart simulation.')
            robot.restart_sim()
            nonlocal_variables["no_touch_count"] = 0
            print('##########restart##########')
        else:
            print('##########continue##########')

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print("no touch count:", nonlocal_variables["no_touch_count"])
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))



if __name__ == '__main__':
    main(utils.get_option())
