# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np
import cPickle as pickle

import signal
import random
import math
import os
import time

from game_ac_network import GameACFFNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
from game_state_eval import GameStateEval

from constants_a3c import ACTION_SIZE
from constants_a3c import PARALLEL_SIZE
from constants_a3c import INITIAL_ALPHA_LOW
from constants_a3c import INITIAL_ALPHA_HIGH
from constants_a3c import INITIAL_ALPHA_LOG_RATE
from constants_a3c import MAX_TIME_STEP
from constants_a3c import CHECKPOINT_DIR
from constants_a3c import LOG_FILE
from constants_a3c import RMSP_EPSILON
from constants_a3c import RMSP_ALPHA
from constants_a3c import GRAD_NORM_CLIP
from constants_a3c import USE_GPU
from constants_a3c import QUERY_SIZE
from constants_a3c import EVAL_STEPS
from constants_a3c import EVAL_NUM


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


device = "/cpu:0"
if USE_GPU:
    device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False

# if USE_LSTM:
#   global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
# else:
global_network = GameACFFNetwork(ACTION_SIZE, QUERY_SIZE, -1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(
    learning_rate=learning_rate_input,
    decay=RMSP_ALPHA,
    momentum=0.0,
    epsilon=RMSP_EPSILON,
    clip_norm=GRAD_NORM_CLIP,
    device=device)

for i in range(PARALLEL_SIZE):
    training_thread = A3CTrainingThread(
        i,
        global_network,
        initial_learning_rate,
        learning_rate_input,
        grad_applier,
        MAX_TIME_STEP,
        device=device)
    training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
    intra_op_parallelism_threads=16))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder("float")
value_input = tf.placeholder("float")
tf.summary.scalar("score", score_input)
tf.summary.scalar("value function", value_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[1])
    print(">>> global step set: ", global_t)
    # set wall time
    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'r') as f:
        wall_t = float(f.read())
else:
    print("Could not find old checkpoint")
    # set wall time
    wall_t = 0.0


def train_function(parallel_index):
    global global_t

    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if stop_requested:
            break
        if global_t > MAX_TIME_STEP:
            break

        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                summary_op, score_input,
                                                value_input)
        global_t += diff_global_t


def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True


def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


train_threads = []
for i in range(PARALLEL_SIZE):
    train_threads.append(threading.Thread(target=train_function, args=(i, )))

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
    t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')

for t in train_threads:
    t.join()

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=global_t)

# Save network
pickle.dump(global_network.get_numpy_vars(sess),
            open("saved_network/my_a3c_network.p", "wb"))

# Starting Eval
game_state_eval = GameStateEval(0)

print("\n \n ---------------- Testing --------------- \n \n")

for j in range(EVAL_NUM):
    # start eval iteration
    game_state_eval.evalStart()
    # start first new game
    game_state_eval.reset()
    # do eval iteration
    for i in range(EVAL_STEPS):
        # perceive
        pi_a_, pi_o_, value_ = global_network.run_policy_and_value(
            sess, game_state_eval.s_t)
        action = choose_action(pi_a_)
        query = choose_action(pi_o_)

        game_state_eval.process(action, query)
        if game_state_eval.terminal:
            game_state_eval.reset()
    # End this Eval iteration
    game_state_eval.evalEnd()

print(str(global_t) + " FINAL STEPS")
print("\n \n ---------------- Testing Done :D  --------------- \n \n")
