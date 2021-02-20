import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
X=np.genfromtxt("train_X_svm.csv",delimiter=",",dtype=np.float64,skip_header=1)
Y=np.genfromtxt("train_Y_svm.csv",delimiter=",",dtype=np.float64)

col_min=np.min(X,axis=0)
col_max=np.max(X,axis=0)

def feature_scaling(X,mn=col_min,mx=col_max):
    return (X-mn)/(mx-mn)


def getdata(X,Y,label):
    X1=feature_scaling(X)
    Y1=np.copy(Y)
    Y1=np.where((Y1==label),1,0)
    return X1,Y1

if __name__=="__main__":
    """
    L=[]
    for i in range(1,4):
        train_X,train_Y=getdata(X,Y,i)
        model=SVC(C=10000,kernel="linear").fit(train_X,train_Y)
        L.append(model)
        
    with open("MODEL_FILE.sav",'wb') as file:
            pickle.dump(L,file)

    """
    train_X=feature_scaling(X)
    train_Y=Y
    params_grid=[{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},{'kernel':['linear'],'C':[1,10,100,1000]}]
    
    model=GridSearchCV(SVC(),params_grid,cv=5)
    model.fit(train_X,train_Y)
    print("BestScore:",model.best_score_)
    print("BestC:",model.best_estimator_.C)
    print("BestKernel:",model.best_estimator_.kernel)
    print("BestGamma:",model.best_estimator_.gamma)
   

    final_model=model.best_estimator_
    with open("MODEL_FILE.sav",'wb') as file:
            pickle.dump(final_model,file)
    print(np.where((final_model.predict(train_X)==1)))
