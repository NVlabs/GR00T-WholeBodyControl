import json
import numpy as np
from termcolor import colored
from copy import deepcopy
import os
import sys
import textwrap


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    From https://stackoverflow.com/a/49677241
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def maybe_log_info(message, log=False, color="yellow", indent=False):
    if log:
        if indent:
            message = textwrap.indent(message, " " * 5)
        print(colored(message, color))


def save_meta(args, asset_path):
    # get args
    meta = vars(args)

    # get relative path of running script
    script_path = os.path.abspath(sys.argv[0])
    path_list = script_path.split("/")
    # idx = len(path_list) - 1 - path_list[::-1].index("robosuite_model_zoo")
    # meta["script"] = "/".join(path_list[idx:])
    meta["cmd"] = "python " + " ".join(sys.argv)

    # save meta data
    with open(os.path.join(asset_path, "meta.json"), "w") as outfile:
        json.dump(meta, outfile, indent=4)
