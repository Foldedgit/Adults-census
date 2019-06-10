import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=names, skipinitialspace = True)

print(df.info())
print(df.sample(20))
dfw = (int((df['fnlwgt'].max() - df['fnlwgt'].min()) / 20))

df = df.drop(columns=['capital-gain', 'capital-loss'])  # drop missing data
df = df.drop(columns=['education-num', 'relationship'])  # drop duplications
pd.plotting.scatter_matrix(df, figsize=(6, 6))
plt.show()

ax = sns.countplot(y="workclass", hue="income", data=df)
plt.show()
ax = sns.countplot(y="education", hue="income", data=df)
plt.show()
ax = sns.countplot(y="marital-status", hue="income", data=df)
plt.show()
ax = sns.countplot(y="occupation", hue="income", data=df)
plt.show()
ax = sns.countplot(y="race", hue="income", data=df)
plt.show()
ax = sns.countplot(y="sex", hue="income", data=df)
plt.show()
ax = sns.countplot(y="native-country", hue="income", data=df)
plt.show()


####################################################
#######################################################   Chi Test
####################################################

ageinv = [17,20,30,40,50,60,70,80,90]
agename = ['17-22', '23-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
df['age'] = pd.cut(df['age'], ageinv, labels= agename)


fwinv = np.arange(df['fnlwgt'].min(), df['fnlwgt'].max()+1, int((df['fnlwgt'].max() - df['fnlwgt'].min()) / 20))
fwname = ['part' + str(i) for i in range(1, 21)]
print('fwinv = ', fwinv)
print(fwname)
df['fnlwgt'] = pd.cut(df['fnlwgt'], fwinv, labels=fwname)

whinv = [0, 10, 20, 30, 40, 50, 9999]
whname = ['0-10', '11-20', '21-30', '31-40', '41-50', '51 < ']
df['hours-per-week'] = pd.cut(df['hours-per-week'], whinv, labels=whname)
#
df['workclass'] = df['workclass'].astype('category')
df['education'] = df['education'].astype('category')
df['marital-status'] = df['marital-status'].astype('category')
df['occupation'] = df['workclass'].astype('category')
df['race'] = df['race'].astype('category')
df['sex'] = df['sex'].astype('category')
df['native-country'] = df['native-country'].astype('category')
df['income'] = df['income'].astype('category')

features = ['age', 'fnlwgt', 'hours-per-week', 'workclass', 'education', 'marital-status', 'occupation', 'race',
            'sex', 'native-country']

print(df.head())
print(df.info())
#
# ************************************************************************ Chi test:

while features:
    feature1 = features[0]
    features.remove(feature1)
    for feature2 in features:
        # if feature1 == feature2:
        #     continue
        cont_tab = pd.crosstab(df[feature1], df[feature2], margins=True)
        # create corresponding expected values
        # print(cont_tab.iloc[0, -1])
        # print(cont_tab.iloc[-1, 0])
        # print(cont_tab.iloc[-1, -1])

        nrows = len(cont_tab.iloc[:, 0])-1
        ncolumns = len(cont_tab.iloc[0, :])-1

        rows = cont_tab.iloc[:, -1].values
        columns = cont_tab.iloc[-1, :].values
        total = rows[-1]
        rows = rows[:-1]
        columns = columns[:-1]

        # print(rows)
        # print(columns)
        # print(total)

        expected = np.zeros([nrows, ncolumns])

        j = 0
        for y in rows:
            i = 0
            for x in columns:
                expected[j, i] = x * y / total
                i = i + 1
            j = j + 1
        # print(expected)

        observed = cont_tab.iloc[:-1, :-1].values
        # print(observed)
        print(feature1, '_', feature2, 'chi test result: ', (((observed - expected) ** 2) / expected).sum())
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cont_tab)


