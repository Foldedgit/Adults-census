import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df = pd.read_csv('train.csv', skipinitialspace=True)

print(df.info())    #Print the data frame information
print(df.sample(20))  #Show 20 rows of the data frame randomly

ent = (len(df.index))  #Number of rows(samples) of dataframe

#Derive missing or noisy portion of data for each feature
for col in df.columns:
    if col == 'income':
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(col, ': missing/noisy portion of data=', len(df[df[col].isnull()]) / ent)
        continue

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(col, ': missing/noisy portion of data=', len(df[df[col].isnull() | df[col] < 1]) / ent)


df = df.drop(columns=['capital-gain', 'capital-loss']) #drop based on missing data
# df.loc[df['native-country'] != 'United-States', 'native-country'] = 'others'  # Attribute Construction
for col in df.columns:
    if (df[col].dtype == np.float64 or df[col].dtype == np.int64):
        plt.figure()
        df[col].plot.hist(grid=True, bins=40, rwidth=0.9, color='#607c8e')
        plt.title(col)
        plt.tight_layout()
        continue
    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.tight_layout()


for col in df.columns:
    if (df[col].dtype == np.float64 or df[col].dtype == np.int64):
        plt.figure()
        df.boxplot(column=[col])
        plt.tight_layout()
        continue
    plt.figure()
    ax = sns.countplot(y=col, data=df)

plt.tight_layout()
plt.show()

