# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACFFNetwork

from constants_a3c import GAMMA
from constants_a3c import LOCAL_T_MAX
from constants_a3c import QUERY_SIZE
from constants_a3c import ENTROPY_BETA

# End of Imports

LOG_INTERVAL = 1000
PERFORMANCE_LOG_INTERVAL = 1000


class A3CTrainingThread(object):
    def __init__(self, thread_index, global_network, initial_learning_rate,
                 learning_rate_input, grad_applier, max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        # if USE_LSTM:
        #   self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
        # else:
        self.local_network = GameACFFNetwork(ACTION_SIZE, QUERY_SIZE,
                                             thread_index, device)

        self.local_network.prepare_loss(ENTROPY_BETA)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss,
                var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(), self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GameState(thread_index)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0.0

        # variable controling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
            self.max_global_time_step - global_time_step
        ) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input,
                      value_input, score, value, global_t):
        summary_str = sess.run(
            summary_op, feed_dict={score_input: score,
                                   value_input: value})
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input,
                value_input):
        states = []
        actions = []
        queries = []
        rewards = []
        values = []

        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        # if USE_LSTM:
        #   start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(LOCAL_T_MAX):
            pi_a_, pi_o_, value_ = self.local_network.run_policy_and_value(
                sess, self.game_state.s_t)
            action = self.choose_action(pi_a_)
            query = self.choose_action(pi_o_)

            states.append(self.game_state.s_t)
            actions.append(action)
            queries.append(query)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi_a={}".format(pi_a_))
                print("pi_o={}".format(pi_o_))
                print(" V={}".format(value_))

            # process game
            self.game_state.process(action, query)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            rewards.append(np.clip(reward, -10, 10))

            self.local_t += 1

            if terminal:
                terminal_end = True
                #print("score={}".format(self.episode_reward))

                self._record_score(sess, summary_writer, summary_op,
                                   score_input, value_input,
                                   self.episode_reward, value_, global_t)
                self.episode_reward = 0.0
                self.game_state.reset()
                # if USE_LSTM:
                #   self.local_network.reset_state()
                break

        R = 0.0
        # if not terminal_end:
        R = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        queries.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_q = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for (ai, qi, ri, si, Vi) in zip(actions, queries, rewards, states,
                                        values):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1
            q = np.zeros([QUERY_SIZE])
            q[qi] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_q.append(q)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        # if USE_LSTM:
        #   batch_si.reverse()
        #   batch_a.reverse()
        #   batch_td.reverse()
        #   batch_R.reverse()
        #
        #   sess.run( self.apply_gradients,
        #             feed_dict = {
        #               self.local_network.s: batch_si,
        #               self.local_network.a: batch_a,
        #               self.local_network.td: batch_td,
        #               self.local_network.r: batch_R,
        #               self.local_network.initial_lstm_state: start_lstm_state,
        #               self.local_network.step_size : [len(batch_a)],
        #               self.learning_rate_input: cur_learning_rate } )
        # else:
        sess.run(
            self.apply_gradients,
            feed_dict={
                self.local_network.s: batch_si,
                self.local_network.a: batch_a,
                self.local_network.o: batch_q,
                self.local_network.td: batch_td,
                self.local_network.r: batch_R,
                self.learning_rate_input: cur_learning_rate
            })

        if (self.thread_index == 0) and (
                self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print(
                "### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".
                format(global_t, elapsed_time, steps_per_sec, steps_per_sec *
                       3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t

        return diff_local_t
