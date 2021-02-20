import numpy as np

def replace_null(X):
    ini_array=X
    
    col_mean=np.nanmean(ini_array,axis=0)
    
    inds=np.where(np.isnan(ini_array))
    ini_array[inds]=np.take(col_mean,inds[1])

    return ini_array

def feature_scaling(X):
    
    col_mean=np.mean(X[:,(2,5)],axis=0)
    col_min=np.min(X[:,(2,5)],axis=0)
    col_max=np.max(X[:,(2,5)],axis=0)
    X[:,2]=(X[:,2]-col_min[0])/(col_max[0]-col_min[0])
    X[:,5]=(X[:,5]-col_min[1])/(col_max[1]-col_min[1])

    return X
    




