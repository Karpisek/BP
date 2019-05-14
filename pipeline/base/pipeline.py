import time
import numpy as np

from queue import Queue, Full, Empty
from threading import Thread
from primitives.enums import Mode

DEFAULT_NUMBER_INPUTS = 1
DEFAULT_QUEUE_SIZE = 20


def is_frequency(seq, divider):
    """
    :param seq: current sequence number
    :param divider: 1 means return true on every sequence number, 2 on every second etc...
    :return:
    """
    return seq % divider == 0


class PipeBlock:
    """
    Base class. Pipe block has x input queues and y outputs connected to other PipeBlocks.
    PipeBlock is meant to run in "infinite loop" while he takes on each step some of the objects in any of input queue.
    And puts computed output to any number of output queues. Used queues are capable of multithreading (thread-safe).
    Provides mechanism for communication between PipeBlock instances.
    """
    pipes = []

    def __init__(self, info, pipe_id, output=None, queue_size=DEFAULT_QUEUE_SIZE, print_fps=False, work_modes=None):
        """
        :param info: reference to instance of program info
        :param pipe_id: id of new created pipe-block. Needs to be unique in project.
        Otherwise communication problems may occur.
        :param output: list of PipeBlock instances
        :param queue_size: size of input queues
        :param print_fps: helper parameter used for indication if info about this block should be printed on console
        :param work_modes: specifies workmodes of created PipeBlock
        """
        self.id = pipe_id
        self._print_fps = print_fps
        self._mode = Mode.CALIBRATION_VP
        self._previous_mode = Mode.CALIBRATION_VP
        self._info = info

        self.seq = 0
        self._input = {}
        self._output = {}

        if work_modes is None:
            work_modes = [Mode.CALIBRATION_VP, Mode.CALIBRATION_CORRIDORS, Mode.DETECTION]

        work_modes.append(Mode.SIGNAL)

        self._work_modes = work_modes

        if output is not None:
            for pipe in output:
                pipe.connect(self, queue_size)
                self._output[pipe.id] = pipe

        PipeBlock.pipes.append(self)

    @property
    def mode(self):
        """
        :return: current work mode
        """

        return self._mode

    def start(self):
        """
        Starts computing
        """
        self._run()

    def _run(self):
        """
        Creates infinite loop, on each iteration calls step() method. Before the loop starts before() method is called
        to make stuff need to be done before computation. After the loop is broken after() method is used to clean up
        and other stuff.
        Loop is braked by EOFError exception.
        """

        frame_counter = 0
        clock = time.time()
        self._before()

        try:
            while True:
                self.seq += 1

                if self._print_fps:
                    frame_counter += 1
                    if frame_counter > 100:
                        print(f"{self.__class__.__name__} FPS: ", 1000 / (((time.time() - clock) / frame_counter) * 1000))
                        print(f"Video input progress: {round((self.seq / self._info.frame_count) * 100, 2)}%")
                        frame_counter = 0
                        clock = time.time()

                self._step(self.seq)

        except EOFError:
            self._after()
            print(f"thread {self.__class__.__name__} finally ended")

    def _before(self):
        """
        Called before the computation. Initialization of some components may be done there.
        """
        raise NotImplementedError

    def _step(self, seq):
        """
        Called on every iteration of PipeBlock comunication.
        :param seq: current sequence number
        """

        raise NotImplementedError

    def _after(self):
        """
        Called after computation. Used for cleaning.
        """

        PipeBlock.pipes.remove(self)

    def _mode_changed(self, new_mode):
        """
        Fired on PipeBlock when computation mode has been changed.
        Restarts current sequence number.

        :param new_mode: new mode
        """

        self.seq = 0

    def send(self, message, pipe_id, block=True):
        """
        Used for sending message to another PipeBlock. If desired block is not connected to this one no message is send.
        Puts message into envelope which contains message and work mode.
        Calls deliver() method on desired receiver instance.

        :param message: any type of message
        :param pipe_id: id of desired receiver
        :param block: if the thread should be blocked until message is delivered
        """

        mode = self._mode

        envelope = mode, message

        try:
            self._output[pipe_id].deliver(envelope, pipe_id=self.id, block=block)
        except KeyError:
            pass

    def deliver(self, envelope, pipe_id: int, block):
        """
        Takes envelope and separates message from mode. If this PipeBlock does not support that mode it throws that
        envelope away, otherwise it puts it in input queue corresponding to sender.

        :param envelope: received envelope
        :param pipe_id: sender id
        :param block: if thread should be blocked until envelope is being put into input queue.
        :return:
        """
        try:
            mode, _ = envelope

            if mode in self._work_modes:
                self._input[pipe_id].put(envelope, block=block)

        except Full:
            return

    def receive(self, pipe_id, block=True):
        """
        Takes next envelope in selected input queue and updates work mode corresponding to mode inside envelope.
        If EOFError class is obrained in message, the computation is broken by

        :param pipe_id: selects input by sender id
        :param block: if thread should be blocked until envelope is being get from input queue.
        :return: received message
        :raise EOFError: signalization used for closing the computation
        """

        try:
            mode, message = self._input[pipe_id].get(block)
            if message is EOFError:
                raise EOFError

            self._update_mode(mode)
            return message

        except Empty:
            return None

    def _update_mode(self, mode):
        """
        Updates the mode of current PipeBlock,
        mode_changed() method is called when the mode is being updated.

        :param mode: new mode
        """

        self._previous_mode = self._mode
        self._mode = mode

        if self._previous_mode != self._mode:
            self._mode_changed(self._mode)

    def connect(self, sender, queue_size):
        """
        Connects this input queue to sender PipeBlock instance

        :param sender:
        :param queue_size:
        """

        self._input[sender.id] = Queue(queue_size)

    def __str__(self):
        return f"{self.__class__.__name__}: {[queue.qsize() for key, queue in self._input.items()]}"


class ThreadedPipeBlock(PipeBlock):
    """
    Multithreaded version of PipeBlock, computation is being run in own thread.
    """

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

    def __init__(self, info, pipe_id, output=None, max_steps=np.inf, work_modes=None, deamon=True):
        """
        :param info: reference to all information about examined video
        :param pipe_id: id of new created PipeBlock, should be unique in project, otherwise errors in communication may
        occur.
        :param output: list of output PipeBlocks instances or it derivations
        :param max_steps: maximum number of computation steps (unused)
        :param work_modes: specifies work modes of this PipeBlock
        :param deamon: sets if thread should be run as daemon
        """

        super().__init__(info=info, pipe_id=pipe_id, output=output, work_modes=work_modes)

        self._thread = Thread(target=self._run)
        self._thread.daemon = deamon
        self._max_steps = max_steps

    def start(self):
        """
        Starts thread execution of run() method
        """

        self._thread.start()

    def _before(self):
        pass

    def _step(self, seq):
        raise NotImplementedError

    def _after(self):
        pass

    def join(self):
        """
        Delegates join() call to thread used by this class.
        """

        self._thread.join()
