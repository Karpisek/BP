import time
from enum import Enum
from queue import Queue, Full, Empty
from threading import Thread

import numpy as np

DEFAULT_NUMBER_INPUTS = 1
DEFAULT_QUEUE_SIZE = 20


def is_frequency(seq, frequency):
    return seq % frequency == 0


class Mode(Enum):
    CALIBRATION = 0,
    DETECTION = 1
    SIGNAL = 3,
    END = 4,


class PipeBlock:
    pipes = []

    def __init__(self, pipe_id, output=None, queue_size=DEFAULT_QUEUE_SIZE, print_fps=False, work_modes=None):
        self.id = pipe_id
        self._print_fps = print_fps
        self._mode = Mode.CALIBRATION
        self._previous_mode = Mode.CALIBRATION

        self._input = {}
        self._output = {}

        if work_modes is None:
            work_modes = [Mode.CALIBRATION, Mode.DETECTION]

        work_modes.append(Mode.SIGNAL)

        self._work_modes = work_modes

        if output is not None:
            for pipe in output:
                pipe.connect(self, queue_size)
                self._output[pipe.id] = pipe

        PipeBlock.pipes.append(self)

    def start(self):
        print(f"normal start {self.__class__.__name__}")
        self._run()

    def _run(self):
        seq = 0
        frame_counter = 0

        clock = time.time()

        self._before()

        try:
            while True:
                seq += 1

                if self._print_fps:
                    frame_counter += 1
                    if frame_counter > 100:
                        print(f" {self.__class__.__name__} FPS: ", 1000 / (((time.time() - clock) / frame_counter) * 1000))
                        frame_counter = 0
                        clock = time.time()

                # clock = time.time()
                self._step(seq)
                # print(f" {self.__class__.__name__} Time: ", time.time() - clock)

        except EOFError:
            self._after()
            print(f"thread {self.__class__.__name__} finally ended")

    def _before(self):
        raise NotImplementedError

    def _step(self, seq):
        raise NotImplementedError

    def _after(self):
        PipeBlock.pipes.remove(self)

    def _mode_changed(self, new_mode):
        raise NotImplementedError

    def send(self, message, pipe_id, block=True):
        mode = self._mode

        envelope = mode, message

        try:
            self._output[pipe_id].deliver(envelope, pipe_id=self.id, block=block)
        except KeyError:
            pass

    def deliver(self, envelope, pipe_id: int, block):
        try:
            mode, _ = envelope

            if mode in self._work_modes:
                self._input[pipe_id].put(envelope, block=block)

        except Full:
            return

    def receive(self, pipe_id, block=True):
        try:
            mode, message = self._input[pipe_id].get(block)
            if message is EOFError:
                raise EOFError

            self._update_mode(mode)
            return message

        except Empty:
            return None

    def _update_mode(self, mode):
        self._previous_mode = self._mode
        self._mode = mode

        if self._previous_mode != self._mode:
            self._mode_changed(self._mode)

    def connect(self, sender, queue_size):
        self._input[sender.id] = Queue(queue_size)

    def __str__(self):
        return f"{self.__class__.__name__}: {[queue.qsize() for key, queue in self._input.items()]}"

    @property
    def mode(self):
        return self._mode


class ThreadedPipeBlock(PipeBlock):

    def _mode_changed(self, new_mode):
        pass

    def __init__(self, pipe_id, output=None, max_steps=np.inf, work_modes=None, deamon=True):
        super().__init__(pipe_id=pipe_id, output=output, work_modes=work_modes)

        self._thread = Thread(target=self._run)
        self._thread.daemon = deamon
        self._max_steps = max_steps

    def start(self):
        print(f"thread start {self.__class__.__name__}")
        self._thread.start()

    def _before(self):
        print(f"before {self.__class__.__name__}")

    def _step(self, seq):
        raise NotImplementedError

    def _after(self):
        print(f"after {self.__class__.__name__}")

    def join(self):
        self._thread.join()
