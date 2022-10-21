
# --- Learning --- #
EPISODES = 1000                                 # number of episodes
STEPS = 50                                      # number of steps in one trial

EPS_START = 0.9                                 # e-greedy thr start value
EPS_END = 0.05                                  # e-greedy thr end value
EPS_DECAY = EPISODES                            # e-greedy thr decay
GAMMA = 0.8                                     # Q-learning discount factor
ALPHA = 0.5                                     # learning rate
LR = 0.001                                      # NN optimizer learning rate
HIDDEN_SIZE = 256                               # NN hidden layer size
BATCH_SIZE = 64                                 # Q-learning batch size

STATES = 4
# 25 (initial) ~ 50 [mm]
OPEN_LEN_FIN_PMAX = 26
# 205 ~ 262 [cm/s] (160 ~ 191 [deg/s])
TARGET_VEL_PMAX = 58
OPEN_LEN_FIN_MIN = 25
TARGET_VEL_MIN = 205                            # [cm/s] (160 [deg/s])
ACTIONS = OPEN_LEN_FIN_PMAX * TARGET_VEL_PMAX
DIVISIONS = 20                                  # number of discre divisions

# SEEDS = 543                                   # random seed
IS_PLOT_PROGRESS = True
IS_SAVE_PROGRESS = True
IS_SAVE_WEIGHT = True
IS_PRINT_STATE = True
LOG_INTERVAL = 1                                # interval for logging
SAVE_WEIGHT_INTERVAL = 20                       # interval for saving weights
AVERAGE_INTERVAL = 20

# target objects, reward functions and cameras that can be used
REWARD_LIST = ['success',
               'success-contact',
               'success-rotation',
               'success-contact-rotation']

CAMERA_LIST = ['rs-d415', 'rs-d435']
GRIPPER_LIST = ['2f140', 'softmatics']

# rates for states
# 0 shows the bottom of placement posture, 1 shows the top of it
# GRASP_POSI_HEIGHT_P = [0., 1.]
GRASP_POSI_HEIGHT_P = [0., 0.3]
# 0 shows the grasp for com,
# -0.5 shows the most front side, 1 shows the most back side of the shape
# GRASP_POSI_SLIDE_P = [-0.5, 0.5]
GRASP_POSI_SLIDE_P = [-0.15, 0.15]
# 0 shows the grasp from above,
# if -0.5, the eef tilts toward the front by half the angle of GRASP_ORIENT_VW
GRASP_ORIENT_P = [-0.5, 0.5]
# 0 the fully closed, 1 that it opens further only by the object width
# GRASP_WIDTH_P = [0., 1.]
GRASP_WIDTH_P = [0., 0.3]

# variation width of parameters
GRASP_ORIENT_VW = 60  # deg

# rates for grasp posi height, grasp posi width, grasp orient, and grasp width
# the ranges are 0 ~ 1., -0.5 ~ 0.5, -0.5 ~ 0.5, and 0. ~ 1. respectively
PP_STATE_EXAMPLES = {
    'animal':  [0., 0., 0., 0.3],
    'bandaid': [0., 0., 0., 0.3],
    'bottle':  [0., 0., 0., 0.3],
    'choco':   [0., 0., 0., 0.3],
    'dvd':     [0., 0., 0., 0.3],
    'kimwipe': [0., 0., 0., 0.3],
    'mate':    [0., 0., 0., 0.3],
    'note':    [0., 0., 0., 0.3],
    'oreo':    [0., 0., 0., 0.3],
    'paste':   [0., 0., 0., 0.3],
    'tepra':   [0., 0., 0., 0.3]}

PT_STATE_EXAMPLES = {
    'animal':  [0., 0., 0., 0.3],
    'bandaid': [0., 0., 0., 0.3],
    'bottle':  [0., 0., 0., 0.3],
    'choco':   [0., 0., 0., 0.3],
    'dvd':     [0., 0., 0., 0.3],
    'kimwipe': [0., 0., 0., 0.3],
    'mate':    [0., 0., 0., 0.3],
    'note':    [0., 0., 0., 0.3],
    'oreo':    [0., 0., 0., 0.3],
    'paste':   [0., 0., 0., 0.3],
    'tepra':   [0., 0., 0., 0.3]}

# --- Experimental settings --- #
# width in x-axis direction, width in y-axis direction, and height [m]
# when the object is placed in the conveyor
OBJECT_INFO = {
    'animal':  {'vertical': 0.135, 'grasp': 0.049, 'height': 0.116},
    'bandaid': {'vertical': 0.093, 'grasp': 0.030, 'height': 0.065},
    'bottle':  {'vertical': 0.100, 'grasp': 0.070, 'height': 0.090},
    'choco':   {'vertical': 0.157, 'grasp': 0.059, 'height': 0.094},
    'dvd':     {'vertical': 0.142, 'grasp': 0.052, 'height': 0.125},
    'kimwipe': {'vertical': 0.130, 'grasp': 0.088, 'height': 0.120},
    'mate':    {'vertical': 0.107, 'grasp': 0.020, 'height': 0.100},
    'note':    {'vertical': 0.075, 'grasp': 0.043, 'height': 0.075},
    'oreo':    {'vertical': 0.134, 'grasp': 0.049, 'height': 0.096},
    'paste':   {'vertical': 0.168, 'grasp': 0.034, 'height': 0.044},
    'tepra':   {'vertical': 0.078, 'grasp': 0.025, 'height': 0.068}}

FRAME_INFO = {
    '766mm': {'frame_origin': [[-0.100, -1.050, -0.190],
                               [0.0, 0.0, 0.0]],
              'box_origin':   [[-0.100, -0.900, -0.190],
                               [0.0, 0.0, 90.0]]},
    '600mm': {'frame_origin': [[-0.100, -1.050, -0.356],
                               [0.0, 0.0, 0.0]],
              'box_origin':   [[-0.100, -0.900, -0.356],
                               [0.0, 0.0, 90.0]]},
    '450mm': {'frame_origin': [[-0.100, -1.050, -0.506],
                               [0.0, 0.0, 0.0]],
              'box_origin':   [[-0.100, -0.900, -0.506],
                               [0.0, 0.0, 90.0]]}}

# --- Pybullet simulation --- #
# resetDebugVisualizerCamera(
#   cameraDistance=,
#   cameraYaw=,
#   cameraPitch=,
#   cameraTargetPosition=[0, 0, 0],
#   physicsClientId=id)
VIEWPOINTS_INFO = {
    'back':                     [2.0, 270, -40, [0.2, -0.3, -1.0]],
    'back_conveyor_height':     [2.0, 270, -10, [0.2, 0.3, -0.4]],
    'left':                     [2.5, 180, -40, [0.5, -1.0, -1.0]],
    'left_conveyor_height':     [2.5, 180, -10, [0.5, -1.0, -0.7]],
    'right':                    [2.0, 0, -40, [0.1, 0.0, -1.0]],
    'right_conveyor_height':    [2.0, 0, -10, [0.1, 0.0, -0.5]],
    'front':                    [2.0, 90, -40, [0.3, 0.0, -0.3]],
    'front_conveyor_height':    [2.0, 90, -10, [0.3, 0.8, -0.3]],
    'top':                      [2.0, 90, -70, [0.1, 0.0, -0.1]]}
