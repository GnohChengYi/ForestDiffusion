from ForestDiffusion import ForestDiffusionModel

import matplotlib.pyplot as plt
import numpy as np
import pickle


# load training data
keyframe_path = 'XFC4101/Preparation/keyframes.pkl'
with open(keyframe_path, 'rb') as file:
    keyframes = pickle.load(file)
print('original:', keyframes)
X = np.asarray(keyframes, dtype=np.float32)


# see ForestDiffusion/blob/main/R-Package/ForestDiffusion/man/ForestDiffusion.impute.Rd
# remove some data to simulate missing data
mindex = 50 # index of missing data
X[mindex, 1:3] = (np.nan, np.nan)
print('with missing data:', X)


# reduce n_t and duplicate_K for faster computation; default: n_t=50, duplicate_K=100
nimp = 12
forest_model = ForestDiffusionModel(X, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[0], diffusion_type='vp', n_jobs=-1)
# defaults: impute(k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None)
# X_imps = forest_model.impute(k=nimp)    # regular (fast)


# trial for each r
up_r = 32
diffsqs = np.empty((up_r, X.shape[1]))  # squared differences
for r in range(1, up_r + 1):
    print('r:', r)
    X_imps = forest_model.impute(repaint=True, r=r, j=0.1, k=nimp)   # REPAINT (slow, but better)
    # print('imputations:', X_imps)

    # analysis
    rows = np.empty((nimp, X.shape[1]))
    for i in range(len(X_imps)):
        if nimp == 1:
            # shape of X_imps is different between nimp=1 and nimp>1
            rows[i] = X_imps[mindex]
            break
        rows[i] = X_imps[i][mindex]
    print(f'interest:\n{rows}')
    average = np.average(rows, axis=0)
    print(f'average:\n{average}')
    truth = [mindex, mindex, mindex]
    diffsq = (average - truth)**2    # squared difference
    print(f'diff^2:\n{diffsq}')
    diffsqs[r-1] = diffsq


# visualization
x_axis = range(1, len(diffsqs) + 1)
plt.plot(x_axis, diffsqs)
plt.xlabel('r')
plt.ylabel('Squared Difference')
plt.title('r vs Squared Difference')
plt.legend(['Truth', 'X', 'Y'])
plt.show()





''' experiment with changing number of imputations
averages = np.empty((len(rows), X.shape[1]))
for i in range(len(rows)):
    averages[i] = np.average(rows[:i+1], axis=0)
print(averages)
std_devs = np.std(averages, axis=0) # std_dev of each column of averages
print(std_devs)

# visualization
x_axis = range(1, len(averages) + 1)
colors = ['red', 'blue', 'green']
for i in range(X.shape[1]):
    plt.plot(x_axis, averages[:, i], color=colors[i])
    plt.errorbar(x_axis, averages[:, i], yerr=std_devs[i], fmt='o', capsize=5, color=colors[i])
plt.xlabel('Number of Imputations')
plt.ylabel('Average of Imputations')
plt.title('Average vs Number of Imputations')
plt.legend(['Truth', 'X', 'Y'])
plt.show()
'''