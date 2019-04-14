import queue
import time
from queue import Queue
from threading import Thread

import numpy as np

DEFAULT_NUMBER_INPUTS = 1
DEFAULT_QUEUE_SIZE = 20


def is_frequency(seq, frequency):
    return seq % frequency == 0


class PipeBlock:

    def __init__(self, pipe_id, output=None, queue_size=DEFAULT_QUEUE_SIZE):
        self.id = pipe_id

        self._input = {}
        self._output = {}

        if output is not None:
            for pipe in output:
                pipe.connect(self, queue_size)
                self._output[pipe.id] = pipe

    def start(self):
        raise NotImplementedError

    def _run(self):
        seq = 0
        self._before()

        while True:
            try:
                self._step(seq)
            except EOFError:
                self._after()
                break

    def _before(self):
        raise NotImplementedError

    def _step(self, seq):
        raise NotImplementedError

    def _after(self):
        raise NotImplementedError

    def send(self, message, pipe_id, block=True):
        if message is EOFError:
            self._delegate_end()
            raise EOFError

        else:
            self._output[pipe_id].deliver(message, pipe_id=self.id, block=block)

    def _delegate_end(self):
        for key, output in self._output.items():
            output.deliver(EOFError, pipe_id=self.id, block=False)

    def deliver(self, message, pipe_id: int, block):
        try:
            self._input[pipe_id].put(message, block=block)
        except queue.Full:
            return

    def receive(self, pipe_id, block=True):
        try:
            message = self._input[pipe_id].get(block)
            return message

        except queue.Empty:
            return None

    def connect(self, sender, queue_size):
        self._input[sender.id] = Queue(queue_size)

    def __str__(self):
        return f"{self.__class__.__name__}: {[queue.qsize() for key, queue in self._input.items()]}"


class ThreadedPipeBlock(PipeBlock):


    def __init__(self, pipe_id, output=None, max_steps=np.inf):
        super().__init__(pipe_id, output)

        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._max_steps = max_steps

    def start(self):
        self._thread.start()

    def _run(self):
        # timer = time.time()
        seq = 0
        frame_counter = 0

        self._start()

        while True:
            frame_counter += 1
            seq += 1

            try:
                self._step(seq)
            except EOFError:
                self.end()
                break

            if seq % 100 == 0:
                # print(f"{self.__class__.__name__} FPS: {1000 / (((time.time() - timer) / frame_counter) * 1000)}")

                frame_counter = 0
                # timer = time.time()

        print(f"thread final stop stop: {self.__class__}")

    def _after(self):
        pass

    def _before(self):
        pass

    def end(self):
        self._end()

    def _step(self, seq):
        raise NotImplementedError

    def _start(self):
        print(f"starting thread: {self.__class__}")

    def _end(self):
        print(f"ending thread: {self.__class__}")


class NoOutputError(Exception):
    pass
