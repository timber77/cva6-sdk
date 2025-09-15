import time
import os
from contextlib import redirect_stdout
os.environ["OPENBLAS_USE_HERO"] = "" # Unset here because i dont want any calls during import numpy to openblas/ hero inits
import numpy as np
import sys
import perf
# np.set_printoptions(threshold=np.inf)
np.random.seed(0)
rng = np.random.default_rng(0)
# LOG_PATH = "/scratch/msc25f15/measurements/"
LOG_PATH = "."

DTYPE = np.float64

# SIZES = [(8,8,8),(16,16,16),(32,32,32),(32,48,64),(64,64,64), (128, 128, 128), (256, 256, 256), (512, 512, 512)]
# SIZES = [(32,32,32),(32,64,64), (32,128,128), (32,192,192), (32,256,256), (32,512,512)]
SIZES = [(16,16,16)]
TRANSA = [False]
TRANSB = [False]

REPEAT = 1

class GemmResult:
    def __init__(self,a,b,c_dev,c_host,host_time,device_time,err):
        self.a = a
        self.b = b
        self.c_dev = c_dev
        self.c_host = c_host
        self.host_time = host_time
        self.device_time = device_time
        self.err = err

    def print_success(self):
        if self.err:
            print("RESULT DO NOT MATCH")
        else: print("Result match")
    
    def print_performance(self):
        print("Snitch: %f %u"%(self.device_time, self.device_time))
        print("Host: %f %u"%(self.host_time, self.host_time))
        print(f"Speedup (Dev/Host): {self.host_time/self.device_time:.5f}")

    def print_debug_info(self):
        print(f"a:\n{self.a}")
        print(f"b:\n{self.b}")
        print(f"snitch res:\n{self.c_dev}")
        print(f"host res:\n{self.c_host}")
        print(np.allclose(self.c_dev, self.c_host, rtol=1e-05, atol=1e-08, equal_nan=False))
        print(np.isclose(self.c_dev, self.c_host, rtol=1e-05, atol=1e-08, equal_nan=False))
        wrong_indices = np.argwhere(~np.isclose(self.c_dev, self.c_host, rtol=1e-05, atol=1e-08, equal_nan=False))
        print(wrong_indices)
        for i in range(len(wrong_indices)):
            print(f"m: {wrong_indices[i][0]}, n: {wrong_indices[i][1]}")
            print(f"c_dev: {self.c_dev[wrong_indices[i][0]][wrong_indices[i][1]]}")
            print(f"c_host: {self.c_host[wrong_indices[i][0]][wrong_indices[i][1]]}")


def hero_init():
    # Initialize hero
    print("Initializing hero")
    os.environ["OPENBLAS_USE_HERO"] = "1"
    perf.openblas_hero_init()
    print("Done")
    # input("Press enter to continue")

def gemm_nn(a,b):
    return a @ b
def gemm_nt(a,b):
    return a @ b.T
def gemm_tn(a,b):
    return a.T @ b
def gemm_tt(a,b):
    return a.T @ b.T

def get_gemm_func(trans_a:bool, trans_b:bool) -> callable:
    gemm = gemm_nn
    if trans_a and trans_b:
        gemm = gemm_tt
    if trans_a and not trans_b:
        gemm = gemm_tn
    if not trans_a and trans_b:
        gemm = gemm_nt
    return gemm

# m = rows of res, n = cols of res, k = inner dimension
def gemm(m:int, n:int, k:int, trans_a:bool=False, trans_b:bool=False) -> GemmResult:
    print(f"gemm-{m}-{n}-{k}-ta:{trans_a}-tb:{trans_b}")

    a = rng.random((k,m), dtype=DTYPE) if trans_a else rng.random((m,k), dtype=DTYPE)
    b = rng.random((n,k), dtype=DTYPE) if trans_b else rng.random((k,n), dtype=DTYPE)
    # b = np.ones((n,k), dtype=DTYPE) if trans_b else np.ones((k,n), dtype=DTYPE)

    gemm = get_gemm_func(trans_a, trans_b)


    # Run the gemm on the host
    os.environ["OPENBLAS_USE_HERO"] = ""
    t_start = time.time()
    c_host = gemm(a,b)
    host_time = time.time() - t_start 


    # Run the gemm on the device
    os.environ["OPENBLAS_USE_HERO"] = "1"
    t_start = time.time()
    c_dev = gemm(a,b)
    device_time = time.time()- t_start



    err = not np.allclose(c_dev, c_host, rtol=1e-05, atol=1e-08, equal_nan=False)

    gemm_result = GemmResult(a, b, c_dev, c_host, host_time, device_time, err)
    gemm_result.print_success()
    gemm_result.print_performance()
    return gemm_result


def main():

    # hero_init() # No longer needed as it is inited on loading of .so

    # os.environ["HERO_USE_IOMMU"] = ""

    log_columns_host = ["m", "n", "k", "transa", "transb", "dev_time", "host_time"]
    log_columns_device = ["tot", "dma", "issue", "compute"]
    log_columns = log_columns_host.copy()
    log_columns.extend(log_columns_device)

    host_measurements = []
    logfile = os.path.join(LOG_PATH, f"gemm_{np.dtype(DTYPE).name}.csv")
    with open(logfile, "w") as f:
        f.write(",".join(log_columns) + "\n")

    # gemm(8, 8, 8, False, False) # Warmup gemm. The host is significantly slower on first run (factor ~2x)
    error = 0
    measurement_counter = 0
    # for m_ in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    for m_, n_, k_ in SIZES:
        m = m_
        n = n_
        k = k_
        for trans_a in TRANSA:
            for trans_b in TRANSB:
                for _ in range(REPEAT):
                    measurement_counter += 1
                    gemm_result:GemmResult = gemm(m, n, k, trans_a, trans_b)
                    error += gemm_result.err
                    host_measurements.append((m, n, k, trans_a, trans_b, gemm_result.device_time, gemm_result.host_time))
                    if gemm_result.err:
                        gemm_result.print_debug_info()   

    print("Export timestamps")
    perf.print_perf_cycles_dev(native=False)
    device_cycles = perf.get_perf_cycles_dev()
    # The first 2 are from the init and the third one is the warmup gemm
    dev_cycles_measurements = []
    for dev_cycle in device_cycles[(len(device_cycles)-measurement_counter):]: # only log the actual gemm measurements
        dev_cycles_measurements.append((dev_cycle.tot, dev_cycle.dma, dev_cycle.issue, dev_cycle.compute))
    
    with open(logfile, "a") as f:
        for i, host_measurement in enumerate(host_measurements):
            dev_cycles = dev_cycles_measurements[i]
            f.write(",".join([str(x) for x in (host_measurement + dev_cycles)]) + "\n")


    perf.print_hero_timestamps()

    print(f"Total errors: {error}")
        

if __name__ == "__main__":
    main()
