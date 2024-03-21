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
mindexes = (33,66) # indexes of missing data
X[mindexes, 1:3] = (np.nan, np.nan)
print(f'indexes of missing data: {mindexes}')
# print(f'with missing data:\n{X}')


# reduce n_t and duplicate_K for faster computation; default: n_t=50, duplicate_K=100
forest_model = ForestDiffusionModel(X, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[0], diffusion_type='vp', n_jobs=-1)
nimp = 18  # number of imputations, k: default=1
# defaults: impute(k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None)
# X_imps = forest_model.impute(k=nimp)    # regular (fast)
# X_imps = forest_model.impute(repaint=True, r=5, j=0.4, k=nimp)   # REPAINT (slow, but better)
# print('imputations:', X_imps)


# trial for each r or j
up_10j = 10 # upper bound of 10*j
# up_r = 16  # upper bound of r
diffsqs = np.empty((up_10j, X.shape[1]))  # all sum of squared differences
diffsqs_mindex = np.empty((1, X.shape[1]), dtype=np.float32)  # sum of squared differences for each mindex
for ten_j in range(1, up_10j+1):
    j = ten_j / 10
    print('\nj:', j)
    X_imps = forest_model.impute(repaint=True, j=j, k=nimp)   # REPAINT (slow, but better)
    # print('imputations:', X_imps)
    diffsqs_mindex.fill(0)
    for mindex in mindexes:
        # analysis
        if nimp == 1:
            rows = [X_imps[mindex],]
        else:
            rows = X_imps[:nimp][:, mindex]
        average = np.average(rows, axis=0)
        print(f'average:\n{average}')
        truth = [mindex, mindex, mindex]
        diffsq = (average - truth)**2    # squared difference
        print(f'diff^2:\n{diffsq}')
        diffsqs_mindex += diffsq
    diffsqs[ten_j - 1] = diffsqs_mindex
    print(f'all sum of squared differences:\n{diffsqs}')
print(f'all sum of squared differences:\n{diffsqs}')


std_devs = np.std(diffsqs, axis=0) # std_dev of each column of diffsqs
print(f'standard deviations:\n{std_devs}')


# visualization
x_axis = [ten_j / 10 for ten_j in range(1, len(diffsqs) + 1)]
# x_axis = range(1, up_r + 1)
plt.plot(x_axis, diffsqs)
plt.xlabel('j')
plt.ylabel('Sum of Squared Differences')
plt.title('Sum of Squared Differences vs j')
plt.legend(['Truth', 'X', 'Y'])
plt.show()




'''

# visualization
x_axis = range(1, len(diffsqs) + 1)
colors = ['red', 'blue', 'green']
for i in range(X.shape[1]):
    plt.plot(x_axis, diffsqs[:, i], color=colors[i])
    plt.errorbar(x_axis, diffsqs[:, i], yerr=std_devs[i], fmt='o', capsize=5, color=colors[i])
plt.xlabel('Number of Imputations')
plt.ylabel('Sum of Squared Difference')
plt.title('Sum of Squared Difference vs Number of Imputations')
plt.legend(['Truth', 'X', 'Y'])
plt.show()
'''