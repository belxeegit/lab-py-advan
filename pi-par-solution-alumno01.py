import random
import sys
import time
from multiprocessing import Pool
from numba import njit, prange

# Captura de parámetros
num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
print(f"\n>>> EJECUTANDO CON {num_trials} TRIALS <<<")

def calc_pi_serial(N):
    M = 0
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N

def pi_worker(n_subset):
    m_local = 0
    for _ in range(int(n_subset)):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1.0:
            m_local += 1
    return m_local

def calc_pi_mp(N, n_procs=4):
    chunks = [N // n_procs] * n_procs
    with Pool(processes=n_procs) as pool:
        results = pool.map(pi_worker, chunks)
    return 4 * sum(results) / N

@njit(parallel=True)
def calc_pi_numba(N):
    M = 0
    for i in prange(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N

# --- EJECUCIONES ---
print("\n1. Serial...")
s = time.time()
p_s = calc_pi_serial(num_trials)
print(f"Resultado: {p_s} | Tiempo: {time.time()-s:.4f}s")

print("\n2. Multiprocessing (4 cores)...")
s = time.time()
p_m = calc_pi_mp(num_trials, 4)
print(f"Resultado: {p_m} | Tiempo: {time.time()-s:.4f}s")

print("\n3. Numba Parallel...")
_ = calc_pi_numba(100) # Warmup
s = time.time()
p_n = calc_pi_numba(num_trials)
print(f"Resultado: {p_n} | Tiempo: {time.time()-s:.4f}s")
