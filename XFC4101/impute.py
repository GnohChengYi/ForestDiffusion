from ForestDiffusion import ForestDiffusionModel

import matplotlib.pyplot as plt
import numpy as np
import pickle


# load training data
keyframe_path = 'XFC4101/Preparation/keyframes.pkl'
with open(keyframe_path, 'rb') as file:
    keyframes = pickle.load(file)
print(f'original:\n{keyframes}')
X = np.asarray(keyframes, dtype=np.float32)


# see ForestDiffusion/blob/main/R-Package/ForestDiffusion/man/ForestDiffusion.impute.Rd
# remove some data to simulate missing data
n = len(X)
mratio = 0.1    # ratio of missing data
mindexes = np.random.choice(range(n), size=int(n * mratio), replace=False)
mindexes = np.sort(mindexes)
X[mindexes, 1:3] = np.nan
print(f'indexes of missing data: {mindexes}')
print(f'with missing data:\n{X}')


# reduce n_t and duplicate_K for faster computation; default: n_t=50, duplicate_K=100
forest_model = ForestDiffusionModel(X, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[0], diffusion_type='vp', n_jobs=-1)
nimp = 32  # number of imputations, k: default=1
# defaults: impute(k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None)
X_imps = forest_model.impute(k=nimp)    # regular (fast)
# X_imps = forest_model.impute(repaint=True, r=5, j=0.4, k=nimp)   # REPAINT (slow, but better)
# print('imputations:', X_imps)


# trial for each k or r or j
# up_r = 16  # upper bound of r
# up_10j = 10 # upper bound of 10*j
avg_diffsqs = np.empty((len(X_imps), X.shape[1]))  # averages of squared differences
diffsqs_mindex = np.empty((1, X.shape[1]), dtype=np.float32)  # sum of squared differences for each mindex
for iimp in range(1, len(X_imps)+1):
    # j = ten_j / 10
    print(f'\n{iimp}')
    # X_imps = forest_model.impute(repaint=True, j=j, k=nimp)   # REPAINT (slow, but better), comment out for k
    # print('imputations:', X_imps)
    diffsqs_mindex.fill(0)
    for mindex in mindexes:
        # analysis
        if nimp == 1:
            rows = X_imps[mindex]
            average = rows
        else:
            rows = X_imps[:iimp][:, mindex]
            average = np.average(rows, axis=0)
        print(f'average:\n{average}')
        truth = [mindex, mindex, mindex]
        diffsq = (average - truth)**2    # squared difference
        print(f'diff^2:\n{diffsq}')
        diffsqs_mindex += diffsq
    avg_diffsqs[iimp - 1] = diffsqs_mindex / len(mindexes)
    print(f'average squared differences:\n{avg_diffsqs}')


std_devs = np.std(avg_diffsqs, axis=0) # std_dev of each column of diffsqs
print(f'standard deviations:\n{std_devs}')


# visualization
x_axis = range(1, len(avg_diffsqs) + 1)   # for varying k
# x_axis = range(1, up_r + 1)
# x_axis = [ten_j / 10 for ten_j in range(1, len(avg_diffsqs) + 1)]
colors = ['red', 'blue', 'green']
for i in range(X.shape[1]):
    plt.plot(x_axis, avg_diffsqs[:, i], color=colors[i])
    plt.errorbar(x_axis, avg_diffsqs[:, i], yerr=std_devs[i], fmt='o', capsize=5, color=colors[i])
xlabel = 'k=nimp'
plt.xlabel(xlabel)
plt.ylabel('Average Squared Differences')
plt.title(f'Average Squared Differences vs {xlabel}')
plt.legend(['Truth', 'X', 'Y'])
plt.show()
