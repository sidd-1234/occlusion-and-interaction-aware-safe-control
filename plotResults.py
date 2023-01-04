import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import uniform_filter

p = {}
F = {}
x = {}

for i in range(20):
    p[i+1] = np.genfromtxt(f"/SimData/pedestrianData_{i}.csv", delimiter = ',')
    F[i+1] = np.genfromtxt(f"/SimData/safeProbData_{i}.csv", delimiter = ',')
    x[i+1] = np.genfromtxt(f"/SimData/stateData_{i}.csv", delimiter = ',')


Ts = 1e-2

N = len(F[1])

t = np.arange(N)*Ts

epsilon = 0.125

plt.figure(figsize = (16, 9))

plt.subplot(1, 3, 1)

idx = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10])

for i in idx:
    plt.plot(x[i][:, 0], x[i][:, 1], lw = 2.5)

plt.title("Vehicle Velocity (Interrarrival time)", fontsize = 25)

plt.xlabel("distance in meters", fontsize = 25)

plt.ylabel("speed in m/s", fontsize = 25)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.xlim(0, 250)

plt.grid()

plt.subplot(1, 3, 2)

for i in range(5):
    plt.plot(x[i+11][:, 0], x[i+11][:, 1], lw = 2.5)

plt.title("Vehicle Velocity (Time Horizon)", fontsize = 25)

plt.xlabel("distance in meters", fontsize = 25)

plt.ylabel("speed in m/s", fontsize = 25)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.xlim(0, 250)

plt.grid();

plt.subplot(1, 3, 3)

for i in range(5):
    plt.plot(x[i+21][:, 0], x[i+21][:, 1], lw = 2.5)

plt.title("Vehicle Velocity (Occlusion Size)", fontsize = 25)

plt.xlabel("distance in meters", fontsize = 25)

plt.ylabel("speed in m/s", fontsize = 25)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.xlim(0, 250)

plt.grid()

plt.tight_layout()

plt.figure(figsize = (16, 9))

plt.subplot(1, 3, 1)

for i in idx:
    plt.plot(x[i][:, 0], F[i], lw = 2.5)

plt.axhline(y = 1 - epsilon, lw = 2.5, c = 'tab:blue', label = '$1 - \epsilon$', linestyle = '--')

plt.title("Safe Probability (Interrarrival time)", fontsize = 25)

plt.xlabel("distance in meters", fontsize = 25)

plt.ylabel("Probability", fontsize = 25)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.legend(fontsize = 25)

plt.xlim(0, 250)

plt.grid();

plt.subplot(1, 3, 2)

for i in range(5):
    plt.plot(x[i+11][:, 0], F[i+11], lw = 2.5)

plt.axhline(y = 1 - epsilon, lw = 2.5, c = 'tab:blue', label = '$1 - \epsilon$', linestyle = '--')

plt.title("Safe Probability (Time horizon)", fontsize = 25)

plt.xlabel("distance in meters", fontsize = 25)

plt.ylabel("Probability", fontsize = 25)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.legend(fontsize = 25)

plt.xlim(0, 250)

plt.grid()

plt.subplot(1, 3, 3)

for i in range(5):
    plt.plot(x[i+21][:, 0], F[i+21], lw = 2.5)

plt.axhline(y = 1 - epsilon, lw = 2.5, c = 'tab:blue', label = '$1 - \epsilon$', linestyle = '--')

plt.title("Safe Probability (Occlusion size)", fontsize = 25)

plt.xlabel("distance in meters", fontsize = 25)

plt.ylabel("Probability", fontsize = 25)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.legend(fontsize = 25)

plt.xlim(0, 250)

plt.grid()

plt.tight_layout()

min_safe_speed = np.zeros(20)

idx_full = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

for i in idx_full:
    min_safe_speed[i] = np.min(x[i][:, 1])

plt.figure(figsize = (16, 9))

plt.plot(np.array([5, 10, 15, 20, 25, 30, 35, 40, 45]), min_safe_speed[:5], marker = 'o', lw = 2.5)

plt.title("Minimum Safe Speed v/s Interarrival Time", fontsize = 32)

plt.xlabel("Average Interarrival Time in s", fontsize = 32)

plt.ylabel("Speed in m/2", fontsize = 32)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.grid()

plt.tight_layout()

plt.figure(figsize = (16, 9))

plt.plot(np.array([100, 200, 300, 400, 500])*5e-2, min_safe_speed[5:10], marker = 'o', lw = 2.5)

plt.title("Minimum Safe Speed v/s Preview Horizon", fontsize = 32)

plt.xlabel("Time Horizon in s", fontsize = 32)

plt.ylabel("Speed in m/2", fontsize = 32)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.grid()


plt.figure(figsize = (16, 9))

plt.plot(np.array([0, 0.225, 0.450, 0.675, 0.900]), (min_safe_speed[10:])[::-1], marker = 'o', lw = 2.5)

plt.title("Minimum Safe Speed v/s Occlusion Size", fontsize = 32)

plt.xlabel("Occlusion Size in m", fontsize = 32)

plt.ylabel("Speed in m/2", fontsize = 32)

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.grid()

plt.tight_layout()

plt.show()