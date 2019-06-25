from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import itertools
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import json

def attset(at,xdata,ydata,learner):
    xdata = xdata[at]
    n_folds = 10
    scores = cross_val_score(learner, xdata, ydata, cv=n_folds)
    avscore = np.sum(scores)/n_folds
    print(avscore)
    return avscore

df = pd.read_csv('train.csv', skipinitialspace=True)
df['indicator'] = 'train'
df_test = pd.read_csv('test.csv', skipinitialspace=True)
df_test['income'] = -1
df_test['indicator'] = 'test'

df = df.append(df_test)
df.loc[df['native-country'] != 'United-States', 'native-country'] = 'others'

df = df.drop(columns=['capital-gain', 'capital-loss', 'fnlwgt'])  # drop missing data


eduinv = [0,9,14,140]
eduname = ['1-8', '9-13', '14<']
df['educational-num'] = pd.cut(df['educational-num'], eduinv, labels=eduname)

ageinv = [0,30,40,60,190]
agename = ['17-30', '31-40', '41-60', '61-90']
df['age'] = pd.cut(df['age'], ageinv, labels=agename)

whinv = [0, 39, 41, 999]
whname = ['40 > ', '40', '40 < ']
df['hours-per-week'] = pd.cut(df['hours-per-week'], whinv, labels=whname)

# label encoding
le = LabelEncoder()
stratt = ['race', 'gender', 'occupation', 'relationship', 'workclass', 'native-country','education','marital-status', 'hours-per-week', 'age', 'educational-num']
for item in stratt:
    df[item] = le.fit_transform(df[item])


df_train = df[df['indicator'] == 'train']
df_test = df[df['indicator'] == 'test']

att = ['race', 'gender', 'age', 'occupation', 'educational-num', 'relationship', 'hours-per-week', 'workclass', 'native-country',
         'education','marital-status']



## *************************************************  Find the best combination of features
# X = df_train[att]
# Y = df_train['income']
# learn_algo = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'), sampling_strategy='auto',
#                                        replacement=False, random_state=0,n_jobs=-1)
# record = 0
# restab = pd.DataFrame()
# for j in range(len(att)):
#     comb = tuple(itertools.combinations(att, j+1))
#     comb = np.asanyarray(comb)
#
#     for i in range(len(comb)):
#
#         res = attset(comb[i], X, Y, learn_algo)
#         if res > record:
#             record = res
#             fatt = comb[i]
#         if res > 0.7:
#             tab = pd.DataFrame({'attribute set': [comb[i]], 'score': [res]})
#             restab = restab.append(tab)
#
# export_csv = restab.to_csv(r'TREEattresulttable.csv', index = None, header=True)
#
# print(fatt, record)

fatt = ['race', 'gender', 'age', 'occupation', 'educational-num', 'relationship', 'hours-per-week', 'workclass', 'native-country',
         'education','marital-status']

X = df_train[fatt]
Y = df_train['income']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gridparamtree = [{'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              }]
gridparambag = [{'sampling_strategy': ['majority', 'auto', 'all', 'not minority'],
                 'n_estimators': [2, 3, 4, 5, 6, 8, 10, 13, 16, 21]}]

learn_algo = GridSearchCV(BalancedBaggingClassifier(base_estimator=GridSearchCV(DecisionTreeClassifier(), gridparamtree, cv=5),
                                       replacement=False, random_state=0, n_jobs=-1), gridparambag, cv=5, scoring='roc_auc',
                   error_score='raise', n_jobs=-1)

Y_score = learn_algo.fit(X_train, Y_train).predict(X_test)
print(learn_algo.best_params_)
print(learn_algo.best_estimator_)

with open('besttree.txt', 'w') as file:
    file.write(json.dumps(learn_algo.best_params_))

# Compute ROC curve and ROC area for each class : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(Y_test, Y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("Confusion Matrix")
print(confusion_matrix(Y_test, Y_score))

print("Accuracy")
print(accuracy_score(Y_test, Y_score))

print("Summary")
print(classification_report(Y_test, Y_score))

print("roc_auc_score")
print(roc_auc_score(Y_test, Y_score))


# export for Kaggel competition
learn_algo.fit(X, Y)
Y_testpred = learn_algo.predict(df_test[fatt])
np.savetxt('imbDTresult.csv', np.dstack((np.arange(1, Y_testpred.size+1), Y_testpred))[0],
           delimiter=",", fmt='%i', header="idx,income")

