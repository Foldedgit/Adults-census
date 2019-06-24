import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv', skipinitialspace=True)
df['income'] = df['income'].astype('category')
df.loc[df['native-country'] != 'United-States', 'native-country'] = 'others'  # Attribute Construction
# ****************************   Chi Test

features = [col for col in df.columns if df[col].dtype != np.float64 and df[col].dtype != np.int64]

while features:
    feature1 = features[0]
    features.remove(feature1)
    for feature2 in features:
        cont_tab = pd.crosstab(df[feature1], df[feature2], margins=True)

        nrows = len(cont_tab.iloc[:, 0])-1
        ncolumns = len(cont_tab.iloc[0, :])-1

        rows = cont_tab.iloc[:, -1].values
        columns = cont_tab.iloc[-1, :].values
        total = rows[-1]
        rows = rows[:-1]
        columns = columns[:-1]

        expected = np.zeros([nrows, ncolumns])

        j = 0
        for y in rows:
            i = 0
            for x in columns:
                expected[j, i] = x * y / total
                i = i + 1
            j = j + 1

        observed = cont_tab.iloc[:-1, :-1].values

        print(feature1, '_', feature2, 'chi test result: ', (((observed - expected) ** 2) / expected).sum())
