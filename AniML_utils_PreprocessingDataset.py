import pandas as pd
import numpy as np

# Resampling
def BalanceXy(X,y,method,zeros_to_ones=1):
    if method == 'Upsample':
        print("Upsampling...")
        from sklearn.utils import resample
        X0 = X[y==0]
        X1 = X[y==1]
        # upsample minority
        X1_upsampled = resample(X1,
                                  replace=True, # sample with replacement
                                  n_samples=int(len(X0)*zeros_to_ones), # match number in majority class
                                  random_state=27) # reproducible results
        # combine majority and upsampled minority
        X = pd.concat([X0, X1_upsampled])
        y = pd.concat([pd.DataFrame(0*np.ones(np.size(X0,axis=0))),
                       pd.DataFrame(np.ones(np.size(X1_upsampled,axis=0)))])
        y=np.ravel(y)
        print(method + " balancing Done. (Zeros: " + str(np.sum(y == 0)) + "; Ones:" + str(np.sum(y == 1)) + ")")

    elif method == 'Downsample':
        print("Downsampling...")
        from sklearn.utils import resample
        X0 = X[y==0]
        X1 = X[y==1]
        X0_downsampled = resample(X0,
                                  replace=True, # sample with replacement
                                  n_samples=int(len(X1)*zeros_to_ones), # match number in majority class
                                  random_state=27) # reproducible results

        # combine majority and upsampled minority
        X = pd.concat([X0_downsampled, X1])
        y = pd.concat([pd.DataFrame(0*np.ones(np.size(X0_downsampled,axis=0))), pd.DataFrame(np.ones(np.size(X1,axis=0)))])
        y=np.ravel(y)
        print(method + " balancing Done. (Zeros: " + str(np.sum(y == 0)) + "; Ones:" + str(np.sum(y == 1)) + ")")

    elif method == 'SMOTE':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        X,y = smote.fit_resample(X, y)
        print(method + " balancing Done. (Zeros: " + str(np.sum(y == 0)) + "; Ones:" + str(np.sum(y == 1)) + ")")

    elif method == 'Kernel-SMOTE':
        from imblearn.over_sampling import KMeansSMOTE
        ksmote = KMeansSMOTE(sampling_strategy='auto', random_state=42)
        X,y = ksmote.fit_resample(X, y)
        print(method + " balancing Done. (Zeros: " + str(np.sum(y == 0)) + "; Ones:" + str(np.sum(y == 1)) + ")")

    else:
        raise ValueError("Error. Did not choose balance method (method=Resample\Downsample).")
    return X, y