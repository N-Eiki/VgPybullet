#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import os.path as osp

import utils


if __name__ == '__main__':
    args = utils.get_option()
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    robotdir = osp.join(
        osp.dirname(__file__), '../model/robot')

    if args.gripper_type == "2f140":
        robotId = p.loadURDF(
            osp.join(robotdir, "ur5e_with_2f140.urdf"))
        p.resetBasePositionAndOrientation(
            robotId,
            [0.00000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000])
    elif args.gripper_type == "softmatics":
        robotId = p.loadURDF(
            osp.join(robotdir, "ur5e_with_softmatics.urdf"))
        p.resetBasePositionAndOrientation(
            robotId,
            [0.00000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000])

    while True:
        p.stepSimulation()
