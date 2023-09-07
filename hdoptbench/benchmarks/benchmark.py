from abc import ABC


class Benchmark(ABC):
    name = "Benchmark name"

    def __init__(self):
        super().__init__()
        self._bounds = None
        self._ndim = None
        self.f_global = None
        self.x_global = None
        self.f_shift = None
        self.f_bias = None
        self.support_path = None
        self.verbose = False
        self.n_fe = 0