from queue import Queue

DEFAULT_NUMBER_INPUTS = 1
DEFAULT_QUEUE_SIZE = 20


class PipeBlock:
    _output = []

    def __init__(self, output=None, number_of_inputs=DEFAULT_NUMBER_INPUTS, queue_size=DEFAULT_QUEUE_SIZE):
        self._output = output
        self._input = [Queue(queue_size) for _ in range(number_of_inputs)]

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

    def __str__(self):
        return f"{self.__class__.__name__}: {[q.qsize() for q in self._input]}"


class NoOutputError(Exception):
    pass
