print("Running gemm_simple.py")
import numpy as np
import perf
print("Imports done")
perf.openblas_hero_init()
print("Hero init done")
a = np.random.rand(256, 256).astype(np.float64)
b = np.random.rand(256, 256).astype(np.float64)
c = np.matmul(a, b)
print(c)
print("Done")