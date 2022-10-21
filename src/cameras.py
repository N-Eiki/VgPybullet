#!/usr/bin/env python3

import numpy as np
import pybullet as p


class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    # image_size = (480, 640)
    image_size = (224, 224)
    intrinsics = (450., 0., 320., 0., 450., 240., 0., 0., 1.)

    # Set default camera poses.
    front_position = (1., 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0., 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)

    right_position = (0.65, -0.3, 0.2)
    right_rotation = (np.pi / 6.0, np.pi, np.pi)
    right_rotation = p.getQuaternionFromEuler(right_rotation)
    # above_position = (0.65, 0.0, 0.4)

    above_position = (0.5, 0., 1.0)
    above_rotation = (0.0, np.pi, np.pi/2.0)
    above_rotation = p.getQuaternionFromEuler(above_rotation)

    # Default camera configs.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': front_position,
        'rotation': front_rotation,
        'zrange': (0.01, 10.),      # meter
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': left_position,
        'rotation': left_rotation,
        'zrange': (0.01, 10.),      # meter
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': right_position,
        'rotation': right_rotation,
        'zrange': (0.01, 10.),      # meter
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': above_position,
        'rotation': above_rotation,
        'zrange': (0.01, 10.),      # meter
        'noise': False
    }]


class Oracle():
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
    position = (0.5, 0, 1000.)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (999.7, 1001.),
        'noise': False
    }]
