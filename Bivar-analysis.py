import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.combine import SMOTEENN


df = pd.read_csv('train.csv', skipinitialspace=True)
df = df.drop(columns=['capital-gain', 'capital-loss']) #drop based on missing data
df.loc[df['native-country'] != 'United-States', 'native-country'] = 'others'  # Attribute Construction

fig, ax = plt.subplots()
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(corr, annot=True,vmax=.3, square=True, mask=mask)
ax.set_title("Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()


att = [col for col in df.columns if (df[col].dtype == np.float64 or df[col].dtype == np.int64)]
df2 = df[att]
att = [x for x in att if x != 'income']
enn = SMOTEENN(random_state=0)
x_enn, y_enn = enn.fit_sample(df2[att], df2['income'])
bdf = pd.concat([pd.DataFrame(x_enn), pd.DataFrame(y_enn)], axis=1)
bdf.columns = df2.columns


fig, ax = plt.subplots()
corr = bdf.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(corr, annot=True,vmax=.3, square=True, mask=mask)
ax.set_title("Balanced Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()
