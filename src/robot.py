#!/usr/bin/env python3

import math
import numpy as np
import cv2

from ur5e_logistics import Ur5eTossObjectEnv


class Robot(object):
    """General robot class."""
    def __init__(self,
                renders,
                sim_view,
                urdfdir,
                gripper_type,
                objname,
                framename,
                camera_cfg,
                box_dist,
                save_snapshots,
                num_obj,
                workspace_limits,
                is_testing,
                test_preset_cases, 
                test_preset_file,
                training_target,
                debug=False,
                 ):

        self.workspace_limits = workspace_limits
        self._pixel_size = 0.003125

        self.env = Ur5eTossObjectEnv(
            simView=sim_view,
            urdfRoot=urdfdir,
            gripperType=gripper_type,
            targetObject=objname,
            targetFrame=framename,
            renders=renders,
            cameraConfig=camera_cfg,
            boxDistance=box_dist,
            saveSnapshots=save_snapshots,
            debug=debug)

        if renders:
            self.env.render()
        self.setup_sim_camera()

    def setup_sim_camera(self):
        """Set camera configuration."""
        # self.cam_depth_scale = 0.25
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img, self.bg_seg_mask = \
            self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def restart_sim(self, setStates=None, isMovingObj=None):
        """Resets the simulator environment."""
        return self.env.reset(setStates, isMovingObj)

    def get_camera_data(self):
        """Gets image data from the camera set."""
        color_img, depth_img, seg_mask = self.env.render()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        return color_img, depth_img, seg_mask

    def picktoss(
            self,
            release_width,
            transport_vel,
            throw_vel,
            is_print_state):
        """Executes one pick-and-toss motion."""
        picktoss_results = self.env.picktossstep(
            release_width, transport_vel, throw_vel, is_print_state)
        return picktoss_results

    def pickplace(
            self,
            transport_vel,
            is_print_state):
        """Executes one pick-and-place motion."""
        pickplace_results = self.env.pickplacestep(
            transport_vel, is_print_state)
        return pickplace_results


    def grasp(self, position, heightmap_rotation_angle, workspace_limits, transport_vel):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        print('           tool_rotation (raw) %f [deg]' % np.rad2deg(heightmap_rotation_angle))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = - (heightmap_rotation_angle % np.pi)
        print('           tool_rotation %f [deg]' % np.rad2deg(tool_rotation_angle))

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] , workspace_limits[2][1])

        # Move gripper to location above grasp target
        grasp_location_margin = 0.25
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
        label = self.env.graspstep([tool_rotation_angle, position, location_above_grasp_target,], transport_vel)

        return label

    def check_sim(self):
        return True

    def check_all_grasp(self):
        return len(self.env.ur5e._objectUids)==0
