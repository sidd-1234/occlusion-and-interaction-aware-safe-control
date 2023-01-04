import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import uniform_filter

F_filt = {}

decimation_factor = 2
filter_size = 15

for i in range(20):
    F = np.genfromtxt(f"/CaseData/case{i}.csv", delimiter = ',')
    F_filt[i] = uniform_filter(F[::decimation_factor, ::decimation_factor].T, size = filter_size)
    np.savetxt(f"/SimData/case{i}_filt.csv", F_filt[i], delimiter = ',')


