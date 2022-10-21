#!/usr/bin/env python3

import os
import time
import datetime
import math
import glob
import random


import numpy as np
import pybullet as p
from PIL import Image
import os.path as osp
from collections import namedtuple
from sympy.geometry import Point, Polygon
from config.simparams import OBJECT_INFO, GRASP_ORIENT_VW


class Ur5eToss:
    """Handler for a robotic tossing system placed on the environment."""
    def __init__(self,
                 urdfRootPath=osp.join(osp.dirname(__file__), '../model'),
                 gripperType='2f140',
                 targetObject='bottle',
                 targetFrame='450mm',
                 boxDistance=-0.520,
                 timeStep=1./80.,
                 openLenInitial=0.140,
                 saveSnapshots=False,
                 numObjects=20,
                 isTest=False,
                 debug=False
                 ):

        if debug:
            numObjects = 5

        self._urdfRootPath = urdfRootPath
        self._gripperType = gripperType
        self._targetObject = targetObject
        self._targetFrame = targetFrame
        self._timeStep = timeStep
        self._openLenInitial = openLenInitial
        self._maxForce = 10000.
        self._ur5eEndEffectorIndex = 7
        self._ur5eGripperIndex = 6
        self._p = p
        self._saveSnapshots = saveSnapshots
        self._objectUid = None
        self._boxDistance = boxDistance
        self._numObjects = numObjects
        self._isTest = isTest
        self._blockRandom = 0.2
        self._debug = debug

        orn = self._p.getQuaternionFromEuler([0., 0., np.deg2rad(-90.)])
        # self.ur5eUid = self._p.loadURDF(
        #     osp.join(self._urdfRootPath, "robot", "ur5e_with_"+self._gripperType+".urdf"),
        #     [-0.110, 0.600, -0.280], orn,
        #     useFixedBase=True,
        #     flags=self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.ur5eUid = self._p.loadURDF(
            osp.join(self._urdfRootPath, "robot", "ur5e_with_"+self._gripperType+".urdf"),
            [0, -0.14, 0], orn,
            useFixedBase=True,
            flags=self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        self.numJoints = self._p.getNumJoints(self.ur5eUid)
        for _ in range(80):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

        self.pecuiliar = self.make_jvs(
            0, 0, 0, 0, 0, 0
        )
        self.initial_jvs = self.make_jvs(
            90., -90., 0., 0., 90., 0.)
        self.pt_pregrasp_jvs = self.make_jvs(
            90., -60., 30., 118., 90., 0.)
        self.pt_grasp_jvs = self.make_jvs(
            90., -46., 49., 84., 90., 0.)
        self.pt_way1_jvs = self.make_jvs(
            0., -60., 30., 118., 90., 0.)
        self.pt_throwstart_jvs = self.make_jvs(
            0., -30., 70., 90., 90., 0.)
        self.pt_throwend_jvs = self.make_jvs(
            0., -70., 0., 50., 90., 0.)
        self.pp_pregrasp_jvs = self.make_jvs(
            90., -60., 30., 118., 90., 0.)
        if self._gripperType == "2f140":
            self.pp_grasp_jvs = self.make_jvs(
                90., -46., 49., 84., 90., 0.)
        elif self._gripperType == "softmatics":
            self.pp_grasp_jvs = self.make_jvs(
                90., -48., 47., 86., 90., 0.)
        self.pp_way1_jvs = self.make_jvs(
            0., -60., 30., 118., 90., 0.)
        self.pp_placestart_jvs = self.make_jvs(
            0., -60., 30., 120., 90., 0.)
        if self._gripperType == "2f140":
            self.pp_placeend_jvs = self.make_jvs(
                0., -26., 15., 101., 90., 0.)
        elif self._gripperType == "softmatics":
            self.pp_placeend_jvs = self.make_jvs(
                0., -30., 15., 101., 90., 0.)

        self.parse_joint_info()
        if self._gripperType == "2f140":
            mimic_parent_name = '2f140_joint_finger'
            mimic_children_names = {
                '2f140_base_to_left_outer_knuckle': 1,
                '2f140_base_to_left_inner_knuckle': 1,
                '2f140_base_to_right_inner_knuckle': 1,
                '2f140_left_outer_finger_to_inner': 1,
                '2f140_right_outer_finger_to_inner': 1}
        elif self._gripperType == "softmatics":
            mimic_parent_name = 'knuckle_joint'
            mimic_children_names = {
                'knuckle_joint': 1,
                'second_knuckle_joint': 1,
                'third_knuckle_joint': 1,
                'fourth_knuckle_joint': 1,
                'fifth_knuckle_joint': 1}
        self.setup_mimic_joints(mimic_parent_name, mimic_children_names)

        # Grasped object generation
        # self._objx = 0.48
        # self._objy = 0.74
        self._objx = 0.5
        self._objy = 0.
        self._objHeight = 0.
        self._objOrient = self._p.getQuaternionFromEuler(
            [np.deg2rad(0.), np.deg2rad(0.), np.deg2rad(0.)])

        # Grasping state
        self._objHeightGrasped = OBJECT_INFO[self._targetObject]['height']
        self._objSlideGrasped = OBJECT_INFO[self._targetObject]['vertical']
        self._objWidthGrasped = 0.#OBJECT_INFO[self._targetObject]['grasp']
        self._graspPosiHeight = 0.  # rate 0. ~ 1. * self._objHeightGrasped
        self._graspPosiSlide = 0.  # rate -0.5 ~ 0.5 * self._objSlideGrasped
        self._graspOrient = 0.  # rate -0.5 ~ 0.5 * GRASP_ORIENT_VW
        self._graspWidth = 0.  # rate 0 ~ 1.0 * self._objWidthGrasped

        # Initialize robot poses
        for joint_pose, joint_id in zip(
                self.initial_jvs, self.arm_controllable_joints):
            self._p.resetJointState(self.ur5eUid, joint_id, joint_pose)
            self._p.setJointMotorControl2(self.ur5eUid,
                                          joint_id,
                                          self._p.POSITION_CONTROL,
                                          targetPosition=joint_pose,
                                          force=self._maxForce)

        self.move_gripper(self._openLenInitial)
        for _ in range(160):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

        self._objectUids = []

    def _get_random_object(self, num_objects, test):
        if test:
            urdf_pattern = os.path.join(self._urdfRootPath, 'random_urdfs/*0/*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRootPath, 'random_urdfs/*[1-9]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames


    def randomly_place_objects(self, urdfList):
        if self._debug:
            def tmp():
                indices = [
                    [0.5, 0], #中心
                    [0.5+0.224 , 0.224], 
                    [0.5+0.224 , -0.224],
                    [0.5-0.224, 0.224],
                    [0.5-0.224, -0.224]
                ]
                for ind in indices:
                    yield ind
            indices = tmp()
            objectUids = []
            for urdf_path in urdfList:

                xpos, ypos = indices.__next__()
                angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                orn = self._p.getQuaternionFromEuler([0, 0, angle])
                uid = self._p.loadURDF(urdf_path, [xpos, ypos, 0.3], [orn[0], orn[1], orn[2], orn[3]])
                objectUids.append(uid)
            for _ in range(10):
                    self._p.stepSimulation()
                    time.sleep(self._timeStep)
        else:
            # Randomize positions of each object urdf.
            objectUids = []
            for urdf_path in urdfList:
                xpos = self._objx + self._blockRandom * (random.random() -0.5)
                ypos = self._objy + self._blockRandom * (random.random() - .5)
                angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
                orn = self._p.getQuaternionFromEuler([0, 0, angle])
                uid = self._p.loadURDF(urdf_path, [xpos, ypos, 0.3], [orn[0], orn[1], orn[2], orn[3]])
                objectUids.append(uid)
                time.sleep(0.1)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
                for _ in range(10):
                    self._p.stepSimulation()
                    time.sleep(self._timeStep)
        return objectUids

    def reset(self, setStates=None, isMovingObj=None):
        """Regenerates target object."""
        self.move_gripper(self._openLenInitial)
        for _ in range(40):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

        # Grasped object generation
        # if self._objectUid is not None:
        #     self._p.removeBody(self._objectUid)
        # multiple objects
        for i in self._objectUids:
            self._p.removeBody(i)
        urdfList = self._get_random_object(self._numObjects, self._isTest)
        self._objectUids = self.randomly_place_objects(urdfList)
        self._objectUids_eval = self._objectUids.copy()

        # Converting params to format used to change the grasp config
        if setStates is not None:
            self._graspPosiHeight = setStates[0] * self._objHeightGrasped
            self._graspPosiSlide = setStates[1] * self._objSlideGrasped
            self._graspOrient = setStates[2] * GRASP_ORIENT_VW
            self._graspWidth = setStates[3] * self._objWidthGrasped

        # # Loading object model in simulation
        # if isMovingObj:     # dynamic object
        #     self._objectUid = self._p.loadURDF(osp.join(
        #         self._urdfRootPath,
        #         "object",
        #         self._targetObject+".urdf"),
        #         [self._objx + self._graspPosiSlide,
        #          self._objy - 1.,
        #          -0.2 + self._objHeightGrasped / 2.],
        #         self._objOrient)
        #     posi_x = self._objx + self._graspPosiSlide
        #     posi_z = -0.2 + self._objHeightGrasped / 2. + 0.01
        #     cid = self._p.createConstraint(
        #         self._objectUid,
        #         -1, -1, -1,
        #         self._p.JOINT_FIXED,
        #         [0, 0, 0], [0, 0, 0],
        #         [posi_x, 0, posi_z])
        #     posi = [posi_x, 0, posi_z]
        #     while (self._objy + 0.01) > posi[1]:
        #         print(posi[1])
        #         posi[1] += 0.01
        #         orn = self._p.getQuaternionFromEuler([0, 0, 0])
        #         self._p.changeConstraint(
        #             cid, posi, jointChildFrameOrientation=orn, maxForce=500)
        #         self._p.stepSimulation()
        #         time.sleep(.02)
        #     self._p.removeConstraint(cid)
        # # else:               # static object
        #     self._objectUid = self._p.loadURDF(osp.join(
        #         self._urdfRootPath,
        #         "object",
        #         self._targetObject+".urdf"),
        #         [self._objx + self._graspPosiSlide,
        #          self._objy,
        #          self._objHeight],
        #         self._objOrient)
        for _ in range(10):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

    def parse_joint_info(self):
        """Parses hand joint information."""
        numJoints = self._p.getNumJoints(self.ur5eUid)
        jointInfo = namedtuple(
            'jointInfo',
            ['id', 'name', 'type', 'damping',
             'friction', 'lowerLimit', 'upperLimit',
             'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self._p.getJointInfo(self.ur5eUid, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != self._p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                self._p.setJointMotorControl2(
                    self.ur5eUid,
                    jointID,
                    self._p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0)
            info = jointInfo(
                jointID, jointName, jointType, jointDamping, jointFriction,
                jointLowerLimit, jointUpperLimit,
                jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)
        self.arm_num_dofs = 6
        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = \
            self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = \
            [info.lowerLimit for info in self.joints
             if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = \
            [info.upperLimit for info in self.joints
             if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = \
            [info.upperLimit - info.lowerLimit for info in self.joints
             if info.controllable][:self.arm_num_dofs]

    def setup_mimic_joints(self, mimic_parent_name, mimic_children_names):
        """Setups mimimc joints for the robotiq hand."""
        self.mimic_parent_id = \
            [joint.id for joint in self.joints
             if joint.name == mimic_parent_name][0]

        self.mimic_child_multiplier = \
            {joint.id: mimic_children_names[joint.name]
             for joint in self.joints if joint.name in mimic_children_names}

        if self._gripperType == "2f140":
            for joint_id, multiplier in self.mimic_child_multiplier.items():
                c = self._p.createConstraint(self.ur5eUid, self.mimic_parent_id,
                                            self.ur5eUid, joint_id,
                                            jointType=self._p.JOINT_GEAR,
                                            jointAxis=[0, 1, 0],
                                            parentFramePosition=[0, 0, 0],
                                            childFramePosition=[0, 0, 0])
                self._p.changeConstraint(
                    c, gearRatio=-multiplier, maxForce=100, erp=1)
        elif self._gripperType == "softmatics":
            for joint_id, multiplier in self.mimic_child_multiplier.items():
                c = self._p.createConstraint(self.ur5eUid, self.mimic_parent_id,
                                            self.ur5eUid, joint_id,
                                            jointType=self._p.JOINT_GEAR,
                                            jointAxis=[1, 0, 0],
                                            parentFramePosition=[0, 0, 0],
                                            childFramePosition=[0, 0, 0])
                self._p.changeConstraint(
                    c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_width, target_param=None):
        """Moves gripper with a target parameter."""
        if target_param is None:
            if self._gripperType == "2f140":
                target_param = float(44.0 - open_width * (44.0 / 0.140))
            elif self._gripperType == "softmatics":
                # target_param = float(50.0 - open_width * (50.0 / 0.160))
                target_param = float(44.0 - open_width * (44.0 / 0.160))
                # target_param = 100
        self._p.setJointMotorControl2(
            self.ur5eUid,
            self.mimic_parent_id,
            self._p.POSITION_CONTROL,
            targetPosition=np.deg2rad(target_param),
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def check_throw_success(self):
        """Judges the throwing success or not."""
        throw_success = False

        y_back = self._boxDistance - 0.150
        y_front = self._boxDistance + 0.150
        if self._targetFrame == "766mm":
            front = Polygon(
                (y_back, -0.180),
                (y_front, -0.180),
                (y_front, 0.020),
                (y_back, 0.020))  # yz
            left = Polygon(
                (-0.340, -0.180),
                (0.140, -0.180),
                (0.140, 0.020),
                (-0.340, 0.020))  # xz
        elif self._targetFrame == "600mm":
            front = Polygon(
                (y_back, -0.346),
                (y_front, -0.346),
                (y_front, -0.146),
                (y_back, -0.146))  # yz
            left = Polygon(
                (-0.340, -0.346),
                (0.140, -0.346),
                (0.140, -0.146),
                (-0.340, -0.146))  # xz
        elif self._targetFrame == "450mm":
            front = Polygon(
                (y_back, -0.496),
                (y_front, -0.496),
                (y_front, -0.296),
                (y_back, -0.296))  # yz
            left = Polygon(
                (-0.340, -0.496),
                (0.140, -0.496),
                (0.140, -0.296),
                (-0.340, -0.296))  # xz

        pos, _ = self._p.getBasePositionAndOrientation(self._objectUid)
        if pos[1] <= y_front:  # over the front line of the boxes
            landxz = Point(pos[0], pos[2])
            landyz = Point(pos[1], pos[2])

            # if any block is inside target bin, provide reward.
            if front.encloses_point(landyz) and left.encloses_point(landxz):
                throw_success = True

        return throw_success

    def make_jvs(self, j1, j2, j3, j4, j5, j6):
        """Makes a joint position vector."""
        jvs = np.array([
            np.deg2rad(j1),
            np.deg2rad(j2),
            np.deg2rad(j3),
            np.deg2rad(j4),
            np.deg2rad(j5),
            np.deg2rad(j6)])
        return jvs

    def move_joints(self, jvs, vel, pg=1.0, vg=0.6):
        """Moves with joint positions."""
        for jp, jid in zip(jvs, self.arm_controllable_joints):
            self._p.setJointMotorControl2(
                self.ur5eUid,
                jid,
                self._p.POSITION_CONTROL,
                targetPosition=jp,
                targetVelocity=vel,
                force=self._maxForce,
                maxVelocity=vel,
                positionGain=pg,
                velocityGain=vg)

    def move_eefpose(
            self, pos, orn, vel, maxit=1000, resth=.001, pg=1.0, vg=0.6, tool_angle=None):
        """Moves with endeffector's positions."""
        jvs = np.array(self._p.calculateInverseKinematics(
            self.ur5eUid,
            self._ur5eEndEffectorIndex,
            pos,
            orn,
            maxNumIterations=maxit,
            residualThreshold=resth)[:6])
        
        if tool_angle:
            jvs[-1] = tool_angle
        
        self.move_joints(jvs, vel, pg, vg)

    def stepsim(self, diff_js, vel, steps=None):
        """Steps simulator."""
        if steps is None:
            steps = int(diff_js * 80./vel + 8)
        for _ in range(steps):
            self._p.stepSimulation()
            time.sleep(self._timeStep)

    def calcdiff_stepsim(self, prev, curr, vel, steps=None):
        prev = np.array(prev)
        curr = np.array(curr)
        """Steps simulator after calculating the joint value difference."""
        diff_js = np.sum(np.abs(prev - curr))
        self.stepsim(diff_js, vel)

    def get_curr_jointState(self):
        ret = []
        for i in range(self._ur5eEndEffectorIndex):
            ret.append(self._p.getJointState(self.ur5eUid, jointIndex=i)[0])
        return ret

    def calculate_eefpose(self, isprint=False):
        """Calculates end effector's pose."""
        eef_pos = np.array(self._p.getLinkState(
            self.ur5eUid, self._ur5eEndEffectorIndex)[0])
        eef_orn = np.array(self._p.getLinkState(
            self.ur5eUid, self._ur5eEndEffectorIndex)[1])

        if isprint:
            print("eef's position:")
            print(eef_pos)
            print("eef's orientation:")
            print(eef_orn)

        return eef_pos, eef_orn

    def calculate_joints(self, isprint=False):
        """Calculates joint values."""
        joints = []
        for joint_id in self.arm_controllable_joints:
            jp, jv, jrf, ajmt = self._p.getJointState(
                self.ur5eUid, joint_id)
            joints.append(np.rad2deg(jp))

        if isprint:
            print("joint values:")
            print(joints)

        return joints

    def execute_picktoss(
            self,
            release_width,
            transport_vel,
            throw_vel,
            is_print_state=False):
        """Executes one pick-and-toss motion in the simulator."""

        # picking
        self.move_joints(self.pt_pregrasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.initial_jvs, self.pt_pregrasp_jvs, transport_vel)

        self.move_joints(self.pt_grasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pt_pregrasp_jvs, self.pt_grasp_jvs, transport_vel)

        eef_pos, eef_orn = self.calculate_eefpose(is_print_state)
        eef_pos[0] += 0.10
        eef_pos[2] += self._graspPosiHeight
        self.move_eefpose(eef_pos, eef_orn, transport_vel)
        self.stepsim(None, None, steps=80)

        self.move_gripper(self._graspWidth)
        self.stepsim(None, None, steps=80)

        self.move_joints(self.pt_pregrasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pt_grasp_jvs, self.pt_pregrasp_jvs, transport_vel)

        self.move_joints(self.pt_way1_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pt_pregrasp_jvs, self.pt_way1_jvs, transport_vel)

        # tossing
        self.move_joints(self.pt_throwstart_jvs, transport_vel, 0.8, 0.8)
        self.calcdiff_stepsim(
            self.pt_way1_jvs, self.pt_throwstart_jvs, transport_vel)

        joint_vel_deg = (throw_vel + 0.854) / 0.01819
        if is_print_state:
            print("Throw joint velocity [deg/s]: "+str(joint_vel_deg))
        self.move_joints(
            self.pt_throwend_jvs, np.deg2rad(joint_vel_deg), 0.5, 1.0)
        self.move_gripper(release_width)

        imgs = []
        checkContact = False
        oldCheckContact = False
        is_first_released = False
        is_first_landed = False
        oldEulerObj = None
        degree_stack = 0
        is_second_released = False
        if self._saveSnapshots:
            for _ in range(160):
                camera_img = self._p.getCameraImage(640, 640)
                imgs.append(Image.fromarray(camera_img[2]))
                self._p.stepSimulation()
                time.sleep(self._timeStep)
            now = datetime.datetime.now()
            logdirpath = osp.join(osp.dirname(__file__), '../logs')
            os.makedirs(logdirpath, exist_ok=True)
            gifname = osp.join(
                logdirpath, now.strftime('%Y%m%d_%H%M%S')+'.gif')
            imgs[0].save(
                gifname,
                save_all=True,
                append_images=imgs[1:],
                duration=20,
                loop=0)
        else:
            for _ in range(160):
                eefP, eefQ, pIner, qIner, pURDF, qURDF, linVel, angVel = \
                     self._p.getLinkState(
                         self.ur5eUid,
                         self.mimic_parent_id,
                         computeLinkVelocity=1)
                objP, objQ = self._p.getBasePositionAndOrientation(
                    self._objectUid)

                # check the target object status
                contact = self._p.getContactPoints(bodyA=self._objectUid)
                if not is_first_landed:
                    if is_first_released:
                        eulerObj = self._p.getEulerFromQuaternion(objQ)
                        if oldEulerObj is not None:
                            degree_inc = np.rad2deg(
                                abs(oldEulerObj[0] - eulerObj[0]))
                            if degree_inc >= 180:  # sign changed
                                degree_inc = np.rad2deg(
                                    abs(3.141592 - oldEulerObj[0])) + \
                                    np.rad2deg(abs(-3.141592 - eulerObj[0]))
                            degree_stack += degree_inc
                            if is_print_state:
                                print("\tRotation angle: " +
                                      str(degree_stack)+' [deg].')
                        oldEulerObj = eulerObj
                    checkContact = False
                    if contact == ():
                        if not oldCheckContact:
                            is_first_released = True
                        checkContact = True
                        if is_print_state:
                            print("\tFlying...")
                    if oldCheckContact and not checkContact:
                        is_first_landed = True
                        if is_print_state:
                            print("\tLanded.")
                    oldCheckContact = checkContact
                else:
                    if contact == () and not is_second_released:
                        # print("Seconde contact was detected.")
                        is_second_released = True

                self._p.stepSimulation()
                time.sleep(self._timeStep)

        # generate return values
        throw_results = {}
        throw_results['release_success'] = is_first_released
        throw_results['land_success'] = self.check_throw_success()
        throw_results['type'] = "direct"
        if is_second_released:
            throw_results['type'] = "wall_contact"
        throw_results['rot_deg'] = degree_stack

        # displaying throwing results summary
        if is_print_state:
            print(
                "\t-> release: " +
                str(["success" if throw_results['release_success']
                    else "failure"][0]) +
                ", land: " +
                str(["success" if throw_results['land_success']
                    else "failure"][0]) +
                ", type: " + throw_results['type'] +
                ", rotation angle [deg]: "+str(throw_results['rot_deg'])
                )

        return throw_results

    def execute_pickplace(
            self,
            transport_vel,
            is_print_state=False):
        """Executes one pick-and-place motion in the simulator."""

        # picking
        self.move_joints(self.pp_pregrasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.initial_jvs, self.pp_pregrasp_jvs, transport_vel)


        ## move z
        self.move_joints(self.pp_grasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_pregrasp_jvs, self.pp_grasp_jvs, transport_vel)

        eef_pos, eef_orn = self.calculate_eefpose(is_print_state)
        eef_pos[0] += 0.10
        eef_pos[2] += self._graspPosiHeight
        self.move_eefpose(eef_pos, eef_orn, transport_vel)
        self.stepsim(None, None, steps=80)

        ## grasp
        # self.move_gripper()
        self.move_gripper(self._graspWidth)
        self.stepsim(None, None, steps=80)

        self.move_joints(self.pp_pregrasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_grasp_jvs, self.pp_pregrasp_jvs, transport_vel)

        self.move_joints(self.pp_way1_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_pregrasp_jvs, self.pp_way1_jvs, transport_vel)

        # placing
        self.move_joints(self.pp_placestart_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_way1_jvs, self.pp_placestart_jvs, transport_vel)

        self.move_joints(self.pp_placeend_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_placestart_jvs, self.pp_placeend_jvs, transport_vel)
        if self._gripperType == "2f140":
            release_offset = 0.02
        elif self._gripperType == "softmatics":
            release_offset = 0.05
        self.move_gripper(self._objWidthGrasped + release_offset)
        self.stepsim(None, None, steps=80)

        self.move_joints(self.pp_placestart_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_placeend_jvs, self.pp_placestart_jvs, transport_vel)

        self.move_joints(self.pp_way1_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_placestart_jvs, self.pp_way1_jvs, transport_vel)

        imgs = []
        if self._saveSnapshots:
            for _ in range(160):
                camera_img = self._p.getCameraImage(640, 640)
                imgs.append(Image.fromarray(camera_img[2]))
                self._p.stepSimulation()
                time.sleep(self._timeStep)
            now = datetime.datetime.now()
            logdirpath = osp.join(
                osp.dirname(__file__), '../logs')
            os.makedirs(logdirpath, exist_ok=True)
            gifname = osp.join(
                logdirpath, now.strftime(
                    '%Y%m%d_%H%M%S')+'.gif')
            imgs[0].save(
                gifname,
                save_all=True,
                append_images=imgs[1:],
                duration=20,
                loop=0)
        else:
            for _ in range(160):
                self._p.stepSimulation()
                time.sleep(self._timeStep)

        place_results = None
        return place_results

    def toolPos2JointPoses(self, tool_position, tool_rotation_angle):
        pos = [tool_position[0],tool_position[1],tool_position[2]]
        orn = self._p.getQuaternionFromEuler([0, -math.pi, 0])
        jointPoses = self._p.calculateInverseKinematics(
            self.ur5eUid, self._ur5eEndEffectorIndex, pos, orn,) #self.ll, self.ul, self.jr, self.rp)
        jointPoses = list(jointPoses)
        jointPoses[self._ur5eEndEffectorIndex] = tool_rotation_angle
        jointPoses = tuple(jointPoses)
        
        return jointPoses[:self._ur5eEndEffectorIndex]

    def close_gripper(self, ):
        self.move_gripper(self._graspWidth)
        self.stepsim(None, None, steps=80)

        self.move_joints(self.pp_pregrasp_jvs, transport_vel)
        self.calcdiff_stepsim(
            self.pp_grasp_jvs, self.pp_pregrasp_jvs, transport_vel)

        self.move_joints(self.pp_way1_jvs, transport_vel)
        self.calcdiff_stepsim(
        self.pp_pregrasp_jvs, self.pp_way1_jvs, transport_vel)

    def go_home(self, transport_vel):
        self.move_joints(self.pp_pregrasp_jvs, transport_vel)
        self.calcdiff_stepsim(
                    self.initial_jvs, self.pp_pregrasp_jvs, transport_vel)


    def go_initial(self, transport_vel):
         self.move_joints(self.initial_jvs, transport_vel)
         self.stepsim(None, None, steps=80)
        
    def execute_grasp(self, actions, transport_vel):
        tool_rotation_angle, position, location_above_grasp_target = actions
        #set home position
        self.go_home(transport_vel)
        self.move_gripper(self._openLenInitial)
        for _ in range(40):
            self._p.stepSimulation()
            time.sleep(self._timeStep)
        #approach above object
        eef_pos, eef_orn = self.calculate_eefpose()
        eef_pos = location_above_grasp_target
        # eef_pos[0] = self._objx #
        # eef_pos[1] = self._objy
        # eef_pos[2] = 0.5
        
        self.move_eefpose(eef_pos, eef_orn, transport_vel, tool_angle=tool_rotation_angle)
        self.stepsim(None, None, steps=100)
        
        #approach object
        eef_pos, eef_orn = self.calculate_eefpose()
        eef_pos = position
        # eef_pos[0] = self._objx #
        # eef_pos[1] = self._objy
        # eef_pos[2] = 0.25
        print('execute position')
        print(eef_pos)
        self.move_eefpose(eef_pos, eef_orn, transport_vel, tool_angle=tool_rotation_angle)
        self.stepsim(None, None, steps=100)

        time.sleep(1)
        #grasp
        self.move_gripper(self._graspWidth)
        self.stepsim(None, None, steps=80)
        time.sleep(1)
        #set home position
        self.go_home(transport_vel)
        #[TODO] check grasp
        grasped = False
        for i,obj_uid in enumerate(self._objectUids):
            objP, objQ = self._p.getBasePositionAndOrientation(obj_uid)
            if objP[2]>0.3:
                grasped = True
                self._p.removeBody(obj_uid)
                self._objectUids = self._objectUids[:i] + self._objectUids[i+1:]
                break

        #set go_initial position
        # self.go_initial(transport_vel)

        return grasped        
    