import zmq
import numpy as np
import re
from distutils.util import strtobool

try:
    import signal
except ImportError as ie:
    ie.args += ("No signal module found. Assuming SIGPIPE is okay.", )
    raise

context = None
socket = None


class GameEnv:
    def __init__(self, zmq_port, mode, thread_num):
        global context
        global socket

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:%s" % zmq_port)

        if mode == "Shooter":
            self.actions = [0, 1, 2, 3, 4, 5, 999]  # shooter
            # actions for selecting each entity or stopping
            # Action 5 - ignore all entities
            # Action 999 - take all entities
        else:
            self.actions = [0, 1, 2, 3, 4, 999]  # EMA

    def process_msg(self, msg):
        tmp = re.split('    ', msg)
        state = tmp[0]
        reward = tmp[1]
        terminal = tmp[2]
        return np.array(state.split(', '),
                        float), float(reward), strtobool(terminal)

    def newGame(self):
        self.socket.send("newGame")
        msg = self.socket.recv()
        while msg == None:
            msg = self.socket.recv()
        return self.process_msg(msg)

    def newGameEval(self):
        self.socket.send("newGameEval")
        msg = self.socket.recv()
        while msg == None:
            msg = self.socket.recv()
        return self.process_msg(msg)

    def step(self, action, query):
        self.socket.send(str(action) + " " + str(query))
        msg = self.socket.recv()
        while msg == None:
            msg = self.socket.recv()
        return self.process_msg(msg)

    def evalInit(self):
        self.socket.send("evalInit")
        msg = self.socket.recv()
        try:
            assert (msg == 'done')
        except AssertionError as e:
            e.args += (msg, )
            raise

    def evalStart(self):
        self.socket.send("evalStart")
        msg = self.socket.recv()
        try:
            assert (msg == 'done')
        except AssertionError as e:
            e.args += (msg, )
            raise

    def evalEnd(self):
        self.socket.send("evalEnd")
        msg = self.socket.recv()
        try:
            assert (msg == 'done')
        except AssertionError as e:
            e.args += (msg, )
            raise

    def getActions(self):
        return self.actions
