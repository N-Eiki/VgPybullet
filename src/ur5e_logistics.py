#!/usr/bin/env python3

import gym
import time
import numpy as np
import pybullet as p
import os.path as osp

import cv2

from ur5e import Ur5eToss
from config.simparams import FRAME_INFO, VIEWPOINTS_INFO


class Ur5eTossObjectEnv(gym.Env):
    """Handler for environment including a robotic tossing system."""
    def __init__(self,
                 simView='back',
                 urdfRoot=osp.join(osp.dirname(__file__), '../model'),
                 gripperType='2f140',
                 targetObject='bottle',
                 targetFrame='450mm',
                 renders=False,
                 openLenInitial=0.140,
                 cameraConfig=None,
                 width=224,
                 height=224,
                 boxDistance=-0.520,
                 saveSnapshots=False,
                 debug=False):

        self._timeStep = 1. / 80.
        self._urdfRoot = urdfRoot
        self._gripperType = gripperType
        self._targetObject = targetObject
        self._targetFrame = targetFrame
        self._renders = renders
        self._openLenInitial = openLenInitial
        self._p = p
        self._width = width
        self._height = height
        self._boxDistance = boxDistance

        if self._renders:
            self.cid = self._p.connect(self._p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = self._p.connect(self._p.GUI)
                vp = VIEWPOINTS_INFO[simView]
                self._p.resetDebugVisualizerCamera(
                    vp[0], vp[1], vp[2], vp[3])
        else:
            self.cid = self._p.connect(self._p.DIRECT)
        self._config = cameraConfig

        # OpenGL camera settings
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self._p.getMatrixFromQuaternion(self._config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = self._config['position'] + lookdir
        focal_len = self._config['intrinsics'][0]
        self._znear, self._zfar = self._config['zrange']
        self._view_mat = self._p.computeViewMatrix(
            self._config['position'], lookat, updir)
        fovh = (self._config['image_size'][0] / 2.0) / focal_len
        fovh = 180.0 * float(np.arctan(fovh)) * 2.0 / np.pi
        aspect_ratio = float(self._config['image_size'][1]) / \
            float(self._config['image_size'][0])
        self._proj_mat = self._p.computeProjectionMatrixFOV(
            fovh, aspect_ratio, self._znear, self._zfar)

        # Initialize pybullet simulation settings
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setGravity(0.000, 0.000, -10.00)
        self._p.setTimeStep(self._timeStep)
        self._p.loadURDF(
            osp.join(self._urdfRoot, "env", "plane.urdf"),
            [0.000, 0.000, -1.100])
        # self._p.loadURDF(
        #     osp.join(self._urdfRoot, "equipment", "conveyor.urdf"),
        #     [0.650, -1.000, -0.210])
        self._p.loadURDF(
            osp.join(self._urdfRoot, "equipment", "conveyor.urdf"),
            [0.5, -1.000, -0.015])

        frame_posi = FRAME_INFO[self._targetFrame]['frame_origin'][0]
        frame_posi[1] = self._boxDistance - 0.150
        frame_orn = FRAME_INFO[self._targetFrame]['frame_origin'][1]
        orn = self._p.getQuaternionFromEuler(
            [np.deg2rad(value) for value in frame_orn])
        self._p.loadURDF(
            osp.join(self._urdfRoot,
                     "env",
                     'frame_'+self._targetFrame+".urdf"),
            frame_posi, orn)

        box_posi = FRAME_INFO[self._targetFrame]['box_origin'][0]
        box_posi[1] = self._boxDistance
        box_orn = FRAME_INFO[self._targetFrame]['box_origin'][1]
        orn = self._p.getQuaternionFromEuler(
            [np.deg2rad(value) for value in box_orn])
        self._p.loadURDF(
            osp.join(self._urdfRoot,
                     "env",
                     "cardboard_box.urdf"),
            box_posi, orn)

        orn = self._p.getQuaternionFromEuler([0.0, 0.0, np.deg2rad(-90.0)])
        # self._p.loadURDF(
        #     osp.join(self._urdfRoot, "robot", "ur5e_basemount.urdf"),
        #     [-0.155, 1.160, -0.980], orn)
        # self._p.loadURDF(
        

        for _ in range(20):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

        self.ur5e = Ur5eToss(
            urdfRootPath=self._urdfRoot,
            gripperType=self._gripperType,
            targetObject=self._targetObject,
            targetFrame=self._targetFrame,
            boxDistance=self._boxDistance,
            timeStep=self._timeStep,
            openLenInitial=self._openLenInitial,
            saveSnapshots=saveSnapshots,
            debug=debug)

        for _ in range(20):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

    def reset(self, setStates=None, isMovingObj=None):
        """Resets robot pose and regenerates target object."""
        self.ur5e.reset(setStates, isMovingObj)

    def render(self):
        """Render things in the simulator."""
        _, _, color, depth, segm = self._p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_mat,
            projectionMatrix=self._proj_mat)

        self._random = np.random.RandomState(0)  # TODO
        # Get color image.
        color_image_size = \
            (self._config['image_size'][0], self._config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if self._config['noise']:
            color = np.int32(color)
            color += np.int32(
                self._random.normal(0, 3, self._config['image_size']))
            color = np.uint8(np.clip(color, 0, 255))
        # Get depth image.
        depth_image_size = \
            (self._config['image_size'][0], self._config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (self._zfar + self._znear -
                 (2. * zbuffer - 1.) * (self._zfar - self._znear))
        depth = (2. * self._znear * self._zfar) / depth
        if self._config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        return color, depth, segm

    def picktossstep(
            self,
            release_width,
            transport_vel,
            throw_vel,
            is_print_state=False):
        """Steps simulation for one pick-and-toss motion."""
        picktoss_results = self.ur5e.execute_picktoss(
            release_width, transport_vel, throw_vel, is_print_state)
        return picktoss_results

    def pickplacestep(
            self,
            transport_vel,
            is_print_state=False):
        """Steps simulation for one pick-and-place motion."""
        pickplace_results = self.ur5e.execute_pickplace(
            transport_vel, is_print_state)
        return pickplace_results

    def graspstep(self, actions, transport_vel):
        return self._step_continuous('grasp', actions, transport_vel)

    def _step_continuous(self, method, actions, transport_vel, box_id=None):
        grasp_success = self.ur5e.execute_grasp(actions, transport_vel)        
        return grasp_success