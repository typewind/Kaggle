import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn.svm import SVC
from  sklearn.mixture import GaussianMixture
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout

path="/Users/typewind/kaggle/Data_Science_London_Scikit_learn/"
X_train=pd.read_csv(path+"train.csv",header=None).as_matrix()
Y_train=pd.read_csv(path+"trainLabels.csv",header=None)[0].as_matrix()
X_test=pd.read_csv(path+"test.csv", header=None).as_matrix()

pca12=PCA(n_components=40,whiten=True)
pca12.fit(np.r_[X_train,X_test])
X_train=pca12.transform(X_train)
X_test=pca12.transform(X_test)


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

def GMM_SVC():
    pca = PCA(whiten=True)
    X_all = pca.fit_transform(np.r_[X_train, X_test])
    X_all = X_all[:,:12]
    #X_test = X_test[:,12]
    #X_test=pca.transform(X_test)
    X_test=X_test[:,:12]
    X_train=X_train[:,:12]
    gmm_new=GaussianMixture(n_components=4, covariance_type='full')
    gmm_new.fit(X_all)
    X_test_gmm=gmm_new.predict_proba(X_test)
    X_train_gmm=gmm_new.predict_proba(X_train)
    svm=SVC()
    svm.fit(X_train_gmm,Y_train)
    pred=svm.predict(X_test_gmm)
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
write_csv(pred_list(GMM_SVC()))

# print the result of simple SVC
#write_csv(pred_list(Simple_SVM()))

# print the result of dummy DNN
#write_csv(pred_list((dummyNet())))

