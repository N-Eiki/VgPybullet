#!/usr/bin/env python3

import numpy as np
import pybullet as p

import utils
import cameras
from robot import Robot


def main(args):
    if args.cam_cfg == 'rs-d415':
        cam_cfgs = cameras.RealSenseD415.CONFIG
    cam_cfg = cam_cfgs[3]

    _ = Robot(True,
              args.sim_view,
              np.float32([[], [], []]),
              args.urdf_dir,
              args.gripper_type,
              args.obj,
              args.frame,
              cam_cfg,
              args.box_dist,
              args.save_snapshots)
    while True:
        p.stepSimulation()


if __name__ == '__main__':
    main(utils.get_option())
