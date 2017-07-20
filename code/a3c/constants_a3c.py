# -*- coding: utf-8 -*-

LOCAL_T_MAX = 500  # repeat step size
RMSP_ALPHA = 0.9  # decay parameter for RMSProp
RMSP_EPSILON = 2e-2  # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4  # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2  # log_uniform high limit for learning rate

PARALLEL_SIZE = 16  # parallel thread size

INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.8  # discount factor for rewards
ENTROPY_BETA = 0.01  # entropy regurarlization constant
MAX_TIME_STEP = 1600000
GRAD_NORM_CLIP = 40.0  # gradient norm clipping
USE_GPU = False  # To use GPU, set True
# USE_LSTM = True # True for A3C LSTM, False for A3C FF

PORT = 7000  # Port for thread 0
MODE = "Shooter"  # Mode of IE
NUM_ENTITIES = 4  # Number of entities - 'shooterName','killedNum', 'woundedNum', 'city'
QUERY_SIZE = NUM_ENTITIES + 1
CONTEXT_LENGTH = 3
STATE_SIZE = 4 * NUM_ENTITIES + 1 + 2 * CONTEXT_LENGTH * NUM_ENTITIES  # Size of the state
ACTION_SIZE = NUM_ENTITIES + 3  # action size for IE task

WORD_LIMIT = 1000
STOP_ACTION = NUM_ENTITIES
IGNORE_ALL = STOP_ACTION + 1
ACCEPT_ALL = 999  #arbitrary
EVAL_STEPS = 5000
EVAL_NUM = 50
