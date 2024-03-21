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
X[50, 1:3] = (np.nan, np.nan)
print(X)


# reduce n_t and duplicate_K for faster computation; default: n_t=50, duplicate_K=100
nimp = 32
forest_model = ForestDiffusionModel(X, n_t=50, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[0], diffusion_type='vp', n_jobs=-1)
# defaults: impute(k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None)
X_imps = forest_model.impute(k=nimp)    # regular (fast)
# X_imps = forest_model.impute(repaint=True, r=10, j=5, k=nimp)   # REPAINT (slow, but better)
print('imputations:', X_imps)


# analysis
rows = np.empty((nimp, X.shape[1]))
for i in range(len(X_imps)):
    X_imp = X_imps[i]
    print(X_imp[50])
    rows[i] = X_imp[50]
print(rows)
averages = np.empty((len(rows), X.shape[1]))
for i in range(len(rows)):
    average = np.average(rows[:i+1], axis=0)
    print(f'average of first {i+1} imputations:', average)
    averages[i] = average
print(averages)
x_axis = range(1, len(averages) + 1)
plt.plot(x_axis, averages)
plt.xlabel('Number of Imputations')
plt.ylabel('Average of Imputations')
plt.title('Average vs Number of Imputations')
plt.legend(['Truth', 'X', 'Y'])
plt.show()
