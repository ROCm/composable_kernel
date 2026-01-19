def sinkhorn_knopp_ref(m, max_iter):
    for i in range(max_iter):
        m /= m.sum(axis=0, keepdims=True)
        m /= m.sum(axis=1, keepdims=True)
    return m

def sinkhorn_knopp_mm(m, max_iter):
     sums = np.ones(m.shape[1])
     for i in range(max_iter):
        sums = (1 / sums) @ m
        sums = m @ ( 1 / sums.T)
     return m / sums

if __name__=="__main__":
    import time
    import numpy as np
    REPS = 10000
    sh = (10,10)
    max_iters = 20
    
    t0 = time.time()
    for _ in range(REPS):
        m = np.random.rand(*sh)
        sinkhorn_knopp_ref(m, max_iters)
    t1 = time.time()
    print(f"{t1-t0} seconds for ref")

    t0 = time.time()
    for _ in range(REPS):
        m = np.random.rand(*sh)
        sinkhorn_knopp_mm(m, max_iters)
    t1 = time.time()
    print(f"{t1-t0} seconds for matrix multiply")

    for _ in range(REPS):
        m = np.random.rand(*sh)
        a = sinkhorn_knopp_ref(m, max_iters)
        b = sinkhorn_knopp_mm(m, max_iters)
        if not np.isclose(a, b).all():
            print(a-b)
    print(f"results match")