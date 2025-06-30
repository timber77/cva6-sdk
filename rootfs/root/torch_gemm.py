# Note make sure to use the correct venv
import time
import os

import torch

import perf

torch.manual_seed(0)
# np.set_printoptions(threshold=np.inf)
LOG_PATH = "/scratch/msc25f15/measurements/torch/"

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
        print(torch.allclose(self.c_dev, self.c_host, rtol=1e-05, atol=1e-08, equal_nan=False))
        print(torch.isclose(self.c_dev, self.c_host, rtol=1e-05, atol=1e-08, equal_nan=False))

def hero_init():
    # Initialize hero
    print("Initializing hero")
    os.environ["OPENBLAS_USE_HERO"] = "1"
    a = torch.rand(16, dtype=torch.float64)
    b = torch.rand(16, dtype=torch.float64)
    test = torch.dot(a,b)
    print("Done")

def gemm_nn(a,b):
    return torch.mm(a, b)
def gemm_nt(a,b):
    return torch.mm(a, b.T)
def gemm_tn(a,b):
    return torch.mm(a.T, b)
def gemm_tt(a,b):
    return torch.mm(a.T, b.T)

def get_gemm_func(trans_a:bool, trans_b:bool) -> callable:
    gemm = gemm_nn
    if trans_a and trans_b:
        gemm = gemm_tt
    if trans_a and not trans_b:
        gemm = gemm_tn
    if not trans_a and trans_b:
        gemm = gemm_nt
    return gemm


# m = rows res, n = cols of res, k = inner dimension
def gemm(m:int, n:int, k:int, trans_a:bool=False, trans_b:bool=False):
    print(f"gemm-{m}-{n}-{k}-ta:{trans_a}-tb:{trans_b}")    

    a = torch.rand((k,m), dtype=torch.float64) if trans_a else torch.rand((m,k), dtype=torch.float64)
    b = torch.rand((k,n), dtype=torch.float64) if trans_b else torch.rand((n,k), dtype=torch.float64)

    gemm = get_gemm_func(trans_a, trans_b)

    # Run the gemm on the device
    os.environ["OPENBLAS_USE_HERO"] = "1"
    t_start = time.time()
    c_dev = gemm(a,b)
    device_time = time.time()- t_start

    # Run the gemm on the host
    os.environ["OPENBLAS_USE_HERO"] = ""
    t_start = time.time()
    c_host = gemm(a,b)
    host_time = time.time()- t_start

    err = not torch.allclose(c_dev, c_host, rtol=1e-05, atol=1e-08, equal_nan=False)

    gemm_result = GemmResult(a, b, c_dev, c_host, host_time, device_time, err)
    gemm_result.print_success()
    gemm_result.print_performance()
    return gemm_result


def main():

    hero_init()

    os.environ["HERO_USE_IOMMU"] = ""

    log_columns_host = ["m", "n", "k", "transa", "transb", "dev_time", "host_time"]
    log_columns_device = ["tot", "dma", "issue", "compute"]
    log_columns = log_columns_host.copy()
    log_columns.extend(log_columns_device)

    host_measurements = []
    logfile = os.path.join(LOG_PATH, "gemm.csv")
    with open(logfile, "w") as f:
        f.write(",".join(log_columns) + "\n")

    gemm(8, 8, 8, False, False) # Warmup gemm. The host is significantly slower on first run (factor ~2x)
    error = 0
    # for m_ in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    for m_ in [64]:
        m = m_
        n = m_
        k = m_
        for trans_a in [False, True]:
            for trans_b in [False, True]:
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
    for dev_cycle in device_cycles[2:]:
        dev_cycles_measurements.append((dev_cycle.tot, dev_cycle.dma, dev_cycle.issue, dev_cycle.compute))
    
    with open(logfile, "a") as f:
        for i, host_measurement in enumerate(host_measurements):
            dev_cycles = dev_cycles_measurements[i]
            f.write(",".join([str(x) for x in (host_measurement + dev_cycles)]) + "\n")

    print(f"Total errors: {error}")
        

if __name__ == "__main__":
    main()
