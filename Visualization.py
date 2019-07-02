import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# *********************************************************************************
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
# *********************************************************************************

# Functions
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

df_train = pd.read_csv('train.csv', skipinitialspace=True)
df_test = pd.read_csv('test.csv', skipinitialspace=True)


df_train.loc[df_train['native-country'] != 'United-States', 'native-country'] = 'others'
df_test.loc[df_test['native-country'] != 'United-States', 'native-country'] = 'others'

df_train = df_train.drop(columns=['capital-gain', 'capital-loss', 'fnlwgt'])  # drop missing data
df_test = df_test.drop(columns=['capital-gain', 'capital-loss', 'fnlwgt'])  # drop missing data

eduinv = [0,9,14,140]
eduname = ['1-8', '9-13', '14<']
df_train['educational-num'] = pd.cut(df_train['educational-num'], eduinv, labels=eduname)
df_test['educational-num'] = pd.cut(df_test['educational-num'], eduinv, labels=eduname)

ageinv = [0,30,40,60,190]
agename = ['17-30', '31-40', '41-60', '61-90']
df_train['age'] = pd.cut(df_train['age'], ageinv, labels=agename)
df_test['age'] = pd.cut(df_test['age'], ageinv, labels=agename)

whinv = [0, 39, 41, 999]
whname = ['40 > ', '40', '40 < ']
df_train['hours-per-week'] = pd.cut(df_train['hours-per-week'], whinv, labels=whname)
df_test['hours-per-week'] = pd.cut(df_test['hours-per-week'], whinv, labels=whname)

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
    df_test[intcols] = scale.transform(df_test[intcols])

for item in stratt:
    le.fit(X[item])
    X[item]=le.transform(X[item])
    df_test[item]=le.transform(df_test[item])
# ************************* feature selection


clf = LinearSVC(C=0.001, penalty="l1", dual=False)
clf = clf.fit(X, Y)
model = SelectFromModel(clf, prefit=True)
feature_idx = model.get_support()
feature_name = X.columns[feature_idx]
print(feature_name)

X = model.transform(X)
df_test = model.transform(df_test)

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
X = X + 1000

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
sme = SMOTETomek(random_state=42)
X_train, Y_train = sme.fit_resample(X_train, Y_train)
clf1 = DecisionTreeClassifier(criterion='gini', max_depth=18, min_samples_leaf=9)
clf2 = ComplementNB(alpha=5, norm=True)
clf3 = KNeighborsClassifier(n_neighbors=12, p=1)
clf4 = MultinomialNB(alpha=0.0001, fit_prior=True)
clf5 = NuSVC(kernel='rbf', nu=0.28, gamma='scale')
clf6 = SVC(kernel='rbf', C=100, gamma='auto', cache_size=2000)
models = (clf1, clf3, clf2, clf4, clf5, clf6)
models = (clf.fit(X_train, Y_train) for clf in models)

# title for the plots
titles = ('DT', 'KNN', 'CNB', 'MNB', 'NuSVC', 'SVC')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# *******************************************   change this section to select intrested features

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Projected variable_1')
    ax.set_ylabel('Projected variable_2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()