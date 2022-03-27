import json
import numpy as np
from matplotlib import pyplot as plt

with open('./cough_features.json') as fp:
    cough = json.load(fp)

with open('./non_cough_features.json') as fp:
    non_cough = json.load(fp)

with open('./cough3_features.json') as fp:
    cough3 = json.load(fp)

cout = 1
cough_final = {}
for i in cough.keys():
    cough_final[str(cout)] = cough[str(i)]
    cout += 1
for i in cough3.keys():
    cough_final[str(cout)] = cough3[str(i)]
    cout += 1
cough = cough_final
del cough_final
del cough3
# print(cough["1"])
sc_shapes = []
sf_shapes = []
zcr_shapes = []
for i in cough.keys():
    sc_shapes.append(len(cough[str(i)]["sc"]))
    sf_shapes.append(len(cough[str(i)]["sf"]))
    zcr_shapes.append(len(cough[str(i)]["zcr"]))


def ploting(zcr_shapes, sf_shapes, sc_shapes):
    plt.plot(zcr_shapes)
    plt.show()
    plt.plot(sf_shapes)
    plt.show()
    plt.plot(sc_shapes)
    plt.show()


# ploting(zcr_shapes, sf_shapes, sc_shapes)

sc_mean = int(np.mean(sc_shapes))
zcr_mean = int(np.mean(zcr_shapes))
sf_mean = int(np.mean(sf_shapes))
# print(f"scmean {int(sc_mean)}, sf mean {int(sf_mean)}, zcr_mean {int(zcr_mean)}")
a = [1,2,4,5]
N = 3
feature_selection = [1,1,1,1,1]
def normalize_struct(cough_, feature_selection):   
    def pad_trunk(a, N):
        a = (a + N * [0])[:N]
        return a
    pad_tuk = 420
    sc = bool(feature_selection[0])
    sf = bool(feature_selection[1])
    zcr = bool(feature_selection[2])
    lpc = bool(feature_selection[3])
    mfcc = bool(feature_selection[4])
    for i in cough_.keys():
        cough_[str(i)]["sc"] = pad_trunk(cough_[str(i)]["sc"],pad_tuk)
        cough_[str(i)]["sf"] = pad_trunk(cough_[str(i)]["sf"], pad_tuk)
        cough_[str(i)]["zcr"] = pad_trunk(cough_[str(i)]["sc"], pad_tuk)
        cough_[str(i)]["full"] = []
        if sc:
            cough_[str(i)]["full"] = cough_[str(i)]["full"] + cough_[str(i)]["sc"]
        if sf:
            cough_[str(i)]["full"] = cough_[str(i)]["full"] + cough_[str(i)]["sf"]
        if zcr:
            cough_[str(i)]["full"] = cough_[str(i)]["full"] + cough_[str(i)]["zcr"]
        if lpc:
            cough_[str(i)]["full"] = cough_[str(i)]["full"] + cough_[str(i)]["lpc_compressed"]
        if mfcc:
            cough_[str(i)]["full"] = cough_[str(i)]["full"] + cough_[str(i)]["mfcc_compressed"]
    return cough_
cough = normalize_struct(cough, feature_selection)
non_cough = normalize_struct(non_cough, feature_selection)
fullset_x = []
fullset_y = []
def make_trainset_cough(cough, fullset_x):
    for i in cough.keys():
        x = cough[str(i)]["full"]
        fullset_x.append(x)
        # y = len(x)*[0]
        # fullset_y.append(y)
    return fullset_x

def make_trainset_non_cough(non_cough, fullset_x):
    for i in non_cough.keys():
        x = non_cough[str(i)]["full"]
        fullset_x.append(x)
        # y = len(x)*[1]
        # fullset_y.append(y)
    return fullset_x
fullset_x= make_trainset_cough(cough, fullset_x)
fullset_y.extend(len(cough)*[0])
fullset_x= make_trainset_non_cough(non_cough, fullset_x)
fullset_y.extend(len(non_cough)*[1])

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(fullset_x, fullset_y, test_size=0.33, random_state=1)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)


for i in range(1, 6):
# ===================== svm =====================
    if i==1:
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(X_train, y_train)
#=================== knn =======================
    elif i==2:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
#====================  SGDClassifier l1 ========================
    elif i==3:
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=5)
        clf.fit(X_train, y_train)
#====================  SGDClassifier l2 ========================
    elif i==4:
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        clf.fit(X_train, y_train)
#==================== naive_bayes =====================
    elif i==5:
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(X_train, y_train)
#======================================================
    elif i==6:
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
#================ VotingClassifier ====================
# from sklearn import svm
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import VotingClassifier
# clf1 = svm.SVC()
# clf2 = KNeighborsClassifier(n_neighbors=3)
# clf3 = SGDClassifier(loss="hinge", penalty="l1", max_iter=5)
# clf4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
# clf5 = GaussianNB()
# clf6 = AdaBoostClassifier(n_estimators=100)

# clf = VotingClassifier(
#     estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3),\
#                 ('clf4', clf4), ('clf5', clf5), ('clf6', clf6)],
#     voting='hard')
# clf.fit(X_train, y_train)
#=======================================================

    y_pred = clf.predict(X_test)
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
    from sklearn.metrics import classification_report

    target_names = ['class 0', 'class 1']
# predicted_met = confusion_matrix(y_test, y_pred)
    predicted_met = classification_report(y_test, y_pred, target_names=target_names)
    print(predicted_met)