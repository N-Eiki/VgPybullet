#!/usr/bin/env python3

import time
import numpy as np
import pybullet as p
import pybullet_data
import os.path as osp
import pybullet_utils.bullet_client as bc
from config.simparams import OBJECT_INFO

import utils


def get_quaternion_from_euler(roll, pitch, yaw):
    """Converts an Euler angle to a quaternion."""
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


if __name__ == '__main__':
    args = utils.get_option()
    timestep = 1./80.
    oneview_duration_sec = 3
    camera_distance = 0.4
    viewpoints = ['', '-x', '+y', '+z']
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    position = [0.00000, 0.000000, 0.000000]
    if args.mode == 'check_coordinate':
        p.setGravity(0, 0, 0)
    else:
        # 1cm high from the plane
        position[2] = OBJECT_INFO[args.obj]['height'] / 2.0 + 0.01
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")

    for simview in viewpoints:
        objectdir = osp.join(
            osp.dirname(__file__), '../model/object')
        objectId = p.loadURDF(
            osp.join(objectdir, args.obj + ".urdf"))
        p.resetBasePositionAndOrientation(
            objectId,
            position,
            get_quaternion_from_euler(0.0, 0.0, 0.0))

        if simview == '+x':    # from the robot side
            p.resetDebugVisualizerCamera(
                camera_distance, 90, -1, [0.0, 0.0, 0.0])
        elif simview == '-x':
            p.resetDebugVisualizerCamera(
                camera_distance, 270, -1, [0.0, 0.0, 0.0])
        elif simview == '+y':  # from the conveying direction side
            p.resetDebugVisualizerCamera(
                camera_distance, 180, -1, [0.0, 0.0, 0.0])
        elif simview == '-y':
            p.resetDebugVisualizerCamera(
                camera_distance, 00, -1, [0.0, 0.0, 0.0])
        elif simview == '+z':  # from the vertically upward direction
            p.resetDebugVisualizerCamera(
                camera_distance, 270, -89, [0.0, 0.0, 0.0])
        elif simview == '-z':
            p.resetDebugVisualizerCamera(
                camera_distance, 270, +89, [0.0, 0.0, 0.0])
        else:
            p.resetDebugVisualizerCamera(
                camera_distance, -225, -45,
                [camera_distance, camera_distance, camera_distance])
        print("-> from "+simview)

        for _ in range(int(oneview_duration_sec * (1./timestep))):
            p.stepSimulation()
            time.sleep(timestep)
        if objectId is not None:
            p.removeBody(objectId)
