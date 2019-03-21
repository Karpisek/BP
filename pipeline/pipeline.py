from queue import Queue
from threading import Thread

DEFAULT_NUMBER_INPUTS = 1
DEFAULT_QUEUE_SIZE = 20


class PipeBlock:

    def __init__(self, output=None, queue_size=DEFAULT_QUEUE_SIZE):
        if output is not None:
            for pipe in output:
                pipe.add_input(queue_size)

        self._output = output
        self._input = []

    def start(self):
        raise NotImplementedError

    def send_to_all(self, message, pipe=0):
        if self._output is None:
            raise NoOutputError

        [receiver.deliver(message, pipe=pipe) for receiver in self._output]

    def send_to(self, message, in_pipe=0, out_pipe=0):
        self._output[out_pipe].deliver(message, pipe=in_pipe)

    def deliver(self, message, pipe=0):
        self._input[pipe].put(message)

    def next(self, pipe=0):
        return self._input[pipe].get()

    def next_nowait(self, pipe=0):
        return self._input[pipe].get(False)

    def add_input(self, queue_size):
        self._input.append(Queue(queue_size))

    def __str__(self):
        return f"{self.__class__.__name__}: {[q.qsize() for q in self._input]}"


class ThreadedPipeBlock(PipeBlock):

    def __init__(self, output, args=None):
        super().__init__(output)

        self._thread = Thread(target=self._run)
        self._thread.daemon = True

    def start(self):
        self._thread.start()

    def _run(self):
        seq = 0
        while True:
            seq += 1
            self._step(seq)

    def _step(self, seq):
        raise NotImplementedError


class NoOutputError(Exception):
    pass
