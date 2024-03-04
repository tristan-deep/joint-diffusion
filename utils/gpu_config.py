"""GPU utilities
Author(s): Tristan Stevens, Ben Luijten
"""
import os

# set TF logging level here, any out of {"0", "1", "2"}
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# set visible GPUs here in quotes comma separated. e.g. "0,1,2"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
import subprocess as sp
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf


def get_gpu_memory(verbose=True):
    """Retrieve memory allocation information of all gpus.
    Arguments
        verbose: prints output if True.
    Retuns
        memory_free_values: list of available memory for each gpu in MiB.
    """
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    # only show enabled devices
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        gpus = [int(gpu) for gpu in gpus.split(",")][: len(memory_free_values)]
        print(
            f"{len(memory_free_values) - len(gpus)}/{len(memory_free_values)} "
            "GPUs were disabled"
        )
        memory_free_values = [memory_free_values[gpu] for gpu in gpus]

    if verbose:
        df = df = pd.DataFrame({"memory": memory_free_values})
        df.index.name = "GPU"
        print(df)
    return memory_free_values


def set_gpu_usage(device=None):
    """Select gpu based on gpu_ids argument.
    Args:
        device (str/int/list): gpu number to select. If None, choose gpu based on
            available memory. Can also be a list of integers to select
            multiple gpus. If device is set to: `cpu`, gpu is disabled.
    """
    if device == "cpu":
        print("Setting device to CPU based on config.")
        tf.config.set_visible_devices([], "GPU")
        return

    if isinstance(device, int) or device is None:
        gpu_ids = [device]
    elif isinstance(device, list):
        gpu_ids = device
    else:
        raise ValueError("gpu_ids must be a list or int or `cpu`.")

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if not gpus:
        print("No available GPUs...")
        return
    # some gpus may have been disabled with the CUDA_VISIBLE_DEVICES
    # environment variable
    available_gpu_ids = [int(gpu.name[-1]) for gpu in gpus]
    print(f"{len(available_gpu_ids)} Available GPU(s): {available_gpu_ids}")

    assert len(gpu_ids) <= len(
        available_gpu_ids
    ), "Number of selected gpus cannot be greater than the amount of available gpus"

    try:
        mem = get_gpu_memory()
    except:
        mem = np.zeros(len(available_gpu_ids))

    assert len(mem) == len(available_gpu_ids), (
        "Some GPUs are not seen by Tensorflow (probably ones with too little tensor cores).\n"
        "Please disable them with CUDA_VISIBLE_DEVICES at the top of this script."
    )

    if None in gpu_ids:
        sorted_gpu_ids = np.argsort(mem)[::-1]

        for i, gpu in zip(range(len(gpu_ids)), sorted_gpu_ids):
            if gpu in available_gpu_ids:
                gpu_ids[i] = int(gpu)

        print("GPU will be automatically chosen based on available memory")
    else:
        bad_gpu = set(gpu_ids) - set(available_gpu_ids)
        if bad_gpu:
            raise ValueError(f"GPU {bad_gpu} not available!!")

    for gpu_id in gpu_ids:
        print(
            f"Selected GPU {gpus[gpu_id].name} with {mem[gpu_id]} MiB of memory available"
        )

    tf.config.experimental.set_visible_devices(
        [gpus[gpu_id] for gpu_id in gpu_ids],
        "GPU",
    )

    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        warnings.warn(
            "Please use set_gpu_usage before using and Tensorflow functionality."
        )
        print(e)

    return gpu_ids


if __name__ == "__main__":
    ## Initilialize GPU
    ## Example on how to use gpu config functions
    set_gpu_usage()
