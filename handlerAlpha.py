import sys
sys.path.append('/home/lsm/rec-similarity-main/Sims')
sys.path.append('/home/lsm/rec-similarity-main/Hanlder')
from AlphaDivergence import AlphaDivergenceSimilarity
import numpy as np
import time
from cdsds import CalMetric
import timeit


resultsDict = {}
nppnsp = {}
neighbours0 = [10, 20, 30, 40, 50, 60]
neighbours = [20, 40, 60, 80, 100, 120]
neighbours2 = [40, 80, 120, 160, 200, 240]
neighbours3 = [10, 40, 70, 100, 130, 160]
neighbours53 = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 160, 180, 200, 220, 260, 300]
neighbourst1 = [20, 40, 70, 100, 130, 160]

start = timeit.default_timer()
resultsDict['ndcg'], resultsDict['hr'], resultsDict['novelty'], resultsDict['mae'], resultsDict['rmse'], \
nppnsp['npp'] = CalMetric().Curvecvcalculate(AlphaDivergenceSimilarity, fold=5, neighbours=neighbours53)
# resultsDict['pre'], resultsDict['rec'], resultsDict['f1'] = CalMetric().Curvecvcalculate(AlphaDivergenceSimilarity, fold=5, neighbours=neighbours53)
end = timeit.default_timer()
time_consumed = end - start#计算代码总时间
print('#' * 100)
print('This is Alpha model')
for key, val in resultsDict.items():
    print(key, val)
for key, val in nppnsp.items():
    print(key, val)
print('Saving dictionary to memory......')
np.save('./alpha.npy', resultsDict)
np.save('./alpha.npy', nppnsp)
np.save('./alpha_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)