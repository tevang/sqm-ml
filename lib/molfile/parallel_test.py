import ray
import time

class Parallel_Operations():

    def __init__(self, POOLSIZE):
        print("Using %i threads." % POOLSIZE)
        ray.init(num_cpus=POOLSIZE)
        for i in range(160000): # in reality, here I have an iterator which stops when it reaches the end of file
            _slow_function.remote()

@ray.remote
def _slow_function():
    benchmark()

def primes(n):
    """
    Just a slow function to monitor the real CPU usage.
    """
    if n == 2:
        return [2]
    elif n < 2:
        return []
    s = []
    for i in range(3, n + 1):
        if i % 2 != 0:
            s.append(i)
    mroot = n ** 0.5
    half = (n + 1) / 2 - 1
    i = 0
    m = 3
    while m <= mroot:
        if s[i]:
            j = int( (m * m - 3) / 2 )
            s[j] = 0
            while j < half:
                try:
                    s[j] = 0
                except IndexError:
                    break
                j += m
        i = i + 1
        m = 2 * i + 3
    l = [2]
    for x in s:
        if x:
            l.append(x)
    return l

def benchmark():
    """
        Just a slow function to monitor the real CPU usage.
    """
    for _ in range(4000):
        count = len(primes(10000000))


if __name__ == "__main__":

    POOLSIZE = 8  # the number of threads to use
    Parallel_Operations(POOLSIZE)