import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from matplotlib.colors import Normalize
import json


class MidpointNormalize(Normalize):  # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


df_train = pd.read_csv('train.csv', skipinitialspace=True)
df_train.loc[df_train['native-country'] != 'United-States', 'native-country'] = 'others'
df_train = df_train.drop(columns=['capital-gain', 'capital-loss', 'fnlwgt'])  # drop missing data

eduinv = [0,9,14,140]
eduname = ['1-8', '9-13', '14<']
df_train['educational-num'] = pd.cut(df_train['educational-num'], eduinv, labels=eduname)

ageinv = [0,30,40,60,190]
agename = ['17-30', '31-40', '41-60', '61-90']
df_train['age'] = pd.cut(df_train['age'], ageinv, labels=agename)

whinv = [0, 39, 41, 999]
whname = ['40 > ', '40', '40 < ']
df_train['hours-per-week'] = pd.cut(df_train['hours-per-week'], whinv, labels=whname)

X = df_train[[col for col in df_train.columns if col != 'income']]
Y = df_train['income']
le = LabelEncoder()
ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
scale = StandardScaler()

intcols = [col for col in X.columns if(X[col].dtype == np.float64 or X[col].dtype == np.int64)]
stratt = [col for col in X.columns if(X[col].dtype != np.float64 and X[col].dtype != np.int64)]

if len(X[intcols].columns) > 0:
    scale.fit(X[intcols])
    X[intcols] = scale.transform(X[intcols])

for item in stratt:
    le.fit(X[item])
    X[item]=le.transform(X[item])

# ************************* feature selection
clf = LinearSVC(C=0.001, penalty="l1", dual=False)
clf = clf.fit(X, Y)
model = SelectFromModel(clf, prefit=True)
feature_idx = model.get_support()
feature_name = X.columns[feature_idx]
print(feature_name)
X = model.transform(X)

ohe.fit(X)
X = ohe.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sme = SMOTETomek(random_state=42)
# sme = SMOTEENN(random_state=42)
X_train, Y_train = sme.fit_resample(X_train, Y_train)

alpha = np.arange(2.0e-10, 1, 1)
fit_prior = ['True', 'False']

gridparamMNB = [{'alpha': alpha, 'fit_prior': fit_prior}]
clf = GridSearchCV(estimator=MultinomialNB(), param_grid=gridparamMNB, cv=5,
                   scoring='roc_auc', n_jobs=-1, error_score='raise')
clf.fit(X_train, Y_train)
print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
scores = clf.cv_results_['mean_test_score'].reshape(len(alpha), len(fit_prior))

with open('bestMNB.txt', 'w') as file:
    file.write(json.dumps(clf.best_params_))

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
Y_score = clf.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.figure()
plt.plot(false_positive_rate, true_positive_rate, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('fit_prior')
plt.ylabel('alpha')
plt.colorbar()
plt.xticks(np.arange(len(fit_prior)), fit_prior, rotation=45)
plt.yticks(np.arange(len(alpha)), alpha)
plt.title('Validation accuracy')
plt.show()

print("Confusion Matrix")
print(confusion_matrix(Y_test, Y_score))

print("Accuracy")
print(accuracy_score(Y_test, Y_score))

print("Summary")
print(classification_report(Y_test, Y_score))

print("roc_auc_score")
print(roc_auc_score(Y_test, Y_score))


model_filename = 'MNBmodel.joblib'
joblib.dump(clf, model_filename)


