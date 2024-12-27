# Lazy matplotlib plotting
import os
import numpy as np
import torch as th

# pickling
import pickle

GLOBAL_IMPORTS = []
GLOBAL_PREFIX_FUNCTIONS = []


class global_functions_lazy:
    # Static call method
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, *args, **kwargs):
        global GLOBAL_PREFIX_FUNCTIONS
        GLOBAL_PREFIX_FUNCTIONS.append((self.name, args, kwargs))


lazyrc = global_functions_lazy("rc")
GLOBAL_IMPORTS.append("from matplotlib import pyplot as plt")
GLOBAL_IMPORTS.append("from matplotlib import rc")
GLOBAL_IMPORTS.append("import sys")
GLOBAL_IMPORTS.append("import os")


class pyplot_lazy:
    def __init__(self, GLOBAL_BUFFER=None, prefix=None):
        if GLOBAL_BUFFER is None:
            self.GLOBAL_BUFFER = []
        else:
            self.GLOBAL_BUFFER = GLOBAL_BUFFER
        if prefix is None:
            self.prefix = "plt"
        else:
            self.prefix = prefix
        self.subplotcnt = 0

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            self.GLOBAL_BUFFER.append((self.prefix, name, args, kwargs))

        return wrapper

    def subplots(self, *args, **kwargs):
        i = self.subplotcnt
        self.subplotcnt += 1
        self.GLOBAL_BUFFER.append((f"fig_{i}, ax_{i}=plt", "subplots", args, kwargs))

        fig = pyplot_lazy(self.GLOBAL_BUFFER, prefix=f"fig_{i}")
        ax = pyplot_lazy(self.GLOBAL_BUFFER, prefix=f"ax_{i}")

        return fig, ax

    def sanitize_args(self, args, kwargs):
        safe_args = []
        for arg in args:
            # If tensor or numpy array, convert to list
            if isinstance(arg, th.Tensor):
                safe_args.append(arg.tolist())
            elif isinstance(arg, np.ndarray):
                safe_args.append(arg.tolist())
            else:
                safe_args.append(arg)

        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, th.Tensor):
                safe_kwargs[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                safe_kwargs[key] = value.tolist()
            else:
                safe_kwargs[key] = value

        return safe_args, safe_kwargs

    def savefig(self, savepath, *save_args, **save_kwargs):
        savepath = os.path.abspath(savepath)
        savepath_lazy = savepath + ".lazyplotlib"
        # savedata = []
        # for name, args, kwargs in self.GLOBAL_BUFFER:
        #     savedata.append((name, args, kwargs))

        # Save global buffer as a pickle file
        with open(savepath_lazy, "wb") as f:
            pickle.dump(self.GLOBAL_BUFFER, f)

        # Create python file that can be run to generate the plot
        savepath_py = os.path.abspath(savepath) + ".lazyplotlib.py"
        with open(savepath_py, "w") as f:
            for import_line in GLOBAL_IMPORTS:
                f.write(f"{import_line}\n")

            for name, args, kwargs in GLOBAL_PREFIX_FUNCTIONS:
                args, kwargs = self.sanitize_args(args, kwargs)
                args = [repr(arg) for arg in args]
                kwargs = [f"{key}={repr(value)}" for key, value in kwargs.items()]

                all_args = ", ".join(args + kwargs)
                f.write(f"{name}({all_args})\n")

            # If the savepath exists, exit
            f.write(f"if os.path.exists('{savepath}'):\n")
            # f.write(f"    print('File {savepath} already exists')\n")
            f.write(f"    sys.exit(0)\n")

            for prefix, name, args, kwargs in self.GLOBAL_BUFFER:
                args, kwargs = self.sanitize_args(args, kwargs)
                args = [repr(arg) for arg in args]
                kwargs = [f"{key}={repr(value)}" for key, value in kwargs.items()]

                all_args = ", ".join(args + kwargs)
                f.write(f"{prefix}.{name}({all_args})\n")

            args, kwargs = self.sanitize_args(save_args, save_kwargs)
            args = [repr(arg) for arg in args]
            kwargs = [f"{key}={repr(value)}" for key, value in kwargs.items()]
            all_args = ", ".join(['"' + savepath + '"'] + args + kwargs)
            f.write(f"{self.prefix}.savefig({all_args})\n")

    def __getitem__(self, key):
        return pyplot_lazy(self.GLOBAL_BUFFER, prefix=f"{self.prefix}[{key}]")

    def close(self):
        self.GLOBAL_BUFFER = []


lazyplot = pyplot_lazy()
