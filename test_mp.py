import time
import multiprocessing as mp

def worker(x):
    return x * x

if __name__ == "__main__":
    ctx = mp.get_context("fork")
    pool = ctx.Pool(16)
    
    start = time.time()
    for i in range(100):
        pool.apply_async(worker, (i,)).get()
    end = time.time()
    print(f"Time for 100 pool executions: {end - start:.4f}s")
