import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from  sklearn.mixture import GaussianMixture
import csv
#from keras.models import Sequential
#from keras.layers import Dense, Dropout

path="/Users/typewind/kaggle/Data_Science_London_Scikit_learn/"
X_train=pd.read_csv(path+"train.csv",header=None).as_matrix()
Y_train=pd.read_csv(path+"trainLabels.csv",header=None)[0].as_matrix()
X_test=pd.read_csv(path+"test.csv", header=None).as_matrix()
X_all=np.r_[X_train, X_test]
Y_train=Y_train.ravel()


#pca12=PCA(n_components=40,whiten=True)
#pca12.fit(np.r_[X_train,X_test])
#X_train=pca12.transform(X_train)
#X_test=pca12.transform(X_test)


def write_csv(prediction):
    with open(path+"output_svc.csv",'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(prediction)

def pred_list(pred):
    prediction = []
    prediction.append(["Id", "Solution"])
    for index, p in enumerate(pred):

        if p > 0.5:
            prediction.append([index + 1, 1])
        else:
            prediction.append([index + 1, 0])
    return prediction

def Simple_SVM():
    svm=SVC()
    svm.fit(X_train,Y_train)
    pred=svm.predict(X_test)
    return pred

    #print(pred_list(pred))

def GMM_RF(X_test, X_train):
    lowest_bic = np.infty
    bic = []

    # find the original GMM parmeter
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X_all)
        bic.append(gmm.aic(X_all))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    #print(best_gmm)

    best_gmm.fit(X_all)
    X_train = best_gmm.predict_proba(X_train)
    X_test = best_gmm.predict_proba(X_test)

    rf = RandomForestClassifier()
    # find the best parameter of random forest
    grid_search_rf = GridSearchCV(rf, param_grid=dict(), verbose=3, scoring='accuracy', cv=10).fit(X_train, Y_train)
    rf_best = grid_search_rf.best_estimator_

    rf_best.fit(X_train, Y_train)

    pred = rf_best.predict(X_test)
    return pred



def dummyNet():
    model=Sequential()
    model.add(Dense(40,activation='relu',input_dim=40))
    #model.add(Dropout(0.1))
    model.add(Dense())
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=256,epochs=200)
    pred=model.predict(X_test)
    return pred




# print the result of GMM_SVC
write_csv(pred_list(GMM_RF(X_test, X_train)))

# print the result of simple SVC
#write_csv(pred_list(Simple_SVM()))

# print the result of dummy DNN
#write_csv(pred_list((dummyNet())))
