# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# 
# Jonas Martin <martinjo@student.ethz.ch>

# Some helper functions to retrieve the device performance counters from the OpenBLAS library. (In theory any shared library containing libhero should work.)


import ctypes
from typing import List
# Load the openblas library. This allows us to call functions from the library and get the performance counters.
blaslib = ctypes.CDLL("libopenblas_riscv64_generic-r0.3.27.dev.so")

class PerfCyclesDev(ctypes.Structure):
    _fields_ = [("tot", ctypes.c_uint32),
                ("dma", ctypes.c_uint32),
                ("issue", ctypes.c_uint32),
                ("compute", ctypes.c_uint32)]

blaslib.get_hero_num_perf_cycles_dev.restype = ctypes.c_int
blaslib.get_hero_perf_cycles_dev.restype = ctypes.POINTER(PerfCyclesDev)


def openblas_hero_init():
    blaslib.openblas_hero_init()

def get_perf_cycles_dev() -> List[PerfCyclesDev]:
    """
    Get the performance cycles for the device.
    """
    # Get the number of performance measurements
    num_perf_measurements = blaslib.get_hero_num_perf_cycles_dev()

    # Get the performance cycles
    perf_cycles_dev_arr_ptr = blaslib.get_hero_perf_cycles_dev()

    # Convert the pointer to a list of PerfCyclesDev objects
    perf_cycles_dev_arr = [perf_cycles_dev_arr_ptr[i] for i in range(num_perf_measurements)]

    # Return the performance cycles
    return perf_cycles_dev_arr


def print_perf_cycles_dev(native=False):
    """
    Print the performance cycles for the device.
    If native = True it calls a print functionfrom hero. Else it retrieves them over the getter and prints them using python.
    """
    if native:
        blaslib.hero_print_perf_device_cycles()
        return

    # Get the performance cycles
    perf_cycles_dev_arr = get_perf_cycles_dev()

    # Print the performance cycles
    for perf_cycles_dev in perf_cycles_dev_arr:
        print(f"tot: {perf_cycles_dev.tot}, dma: {perf_cycles_dev.dma}, issue: {perf_cycles_dev.issue}, compute: {perf_cycles_dev.compute}")

def log_perf_cycles_dev(filename: str):
    """
    Log the performance cycles for the device to a file.
    """
    # Get the performance cycles
    perf_cycles_dev_arr = get_perf_cycles_dev()

    # Open the file for writing
    with open(filename, "w") as f:
        # Write the header
        f.write("tot,dma,issue,compute\n")

        # Write the performance cycles
        for perf_cycles_dev in perf_cycles_dev_arr:
            f.write(f"{perf_cycles_dev.tot},{perf_cycles_dev.dma},{perf_cycles_dev.issue},{perf_cycles_dev.compute}\n")

