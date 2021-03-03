import multiprocessing
import time
import os
import concurrent.futures as confu

def work(num,x = 0):
    """thread worker function"""
    for i in range(10000000):
        x += 1
    return x,x

def multi():
    with confu.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(work, x,1) for x in range(8)]
        (done, notdone) = confu.wait(futures)
        for future in confu.as_completed(futures):
            res1,res2 = future.result()
            print(res2)
        

if __name__ == '__main__':
    start = time.time()
    
    multi()

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")