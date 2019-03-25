import queue
import time
from queue import Queue
from threading import Thread

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

    def send(self, message, pipe_id, block=True):
        self._output[pipe_id].deliver(message, pipe_id=self.id, block=block)

    def deliver(self, message, pipe_id: int, block):
        try:
            self._input[pipe_id].put(message, block=block)
        except queue.Full:
            return

    def receive(self, pipe_id, block=True):
        try:
            return self._input[pipe_id].get(block)
        except queue.Empty:
            return None

    def connect(self, sender, queue_size):
        self._input[sender.id] = Queue(queue_size)

    def __str__(self):
        return f"{self.__class__.__name__}: {[queue.qsize() for key, queue in self._input.items()]}"


class ThreadedPipeBlock(PipeBlock):

    def __init__(self, pipe_id, output=None):
        super().__init__(pipe_id, output)

        self._thread = Thread(target=self._run)
        self._thread.daemon = True

    def start(self):
        self._thread.start()

    def _run(self):
        timer = time.time()
        seq = 0
        frame_counter = 0

        while True:
            frame_counter += 1
            seq += 1

            self._step(seq)

            if seq % 100 == 0:
                # print(f"{self.__class__.__name__} FPS: {1000 / (((time.time() - timer) / frame_counter) * 1000)}")

                frame_counter = 0
                timer = time.time()

    def _step(self, seq):
        raise NotImplementedError


class NoOutputError(Exception):
    pass
