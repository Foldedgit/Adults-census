import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"      #Data set link
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']                #Features name
df = pd.read_csv(url, names=names, skipinitialspace = True)  #Form the data set frame

print(df.info())    #Print the data frame information
print(df.sample(20))  #Show 20 rows of the data frame randomly

ent = (len(df.index))  #Number of rows(samples) of dataframe

#Derive missing or noisy portion of data for each feature
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print('age: missing/noisy portion of data=',len(df[df['age'].isnull() | df['age']<1]) / ent)
    print('workclass: missing/noisy portion of data=',len(df[df['workclass'].isnull()]) / ent)
    print('fnlwgt: missing/noist portion of data=',len(df[df['fnlwgt'].isnull() | df['fnlwgt']<1]) / ent)
    print('education: missing/noisy portion of data=',len(df[ df['education'].isnull()]) / ent)
    print('edu number: missing/noisy portion of data=',len(df[df['education-num'].isnull() | df['education-num']<1]) / ent)
    print('marital status: missing/noisy portion of data=',len(df[df['marital-status'].isnull()]) / ent)
    print('occupation: missing/noisy portion of data=',len(df[df['occupation'].isnull()]) / ent)
    print('relationship: missing/noisy portion of data=',len(df[df['relationship'].isnull()]) / ent)
    print('race: missing/noisy portion of data=',len(df[df['race'].isnull()]) / ent)
    print('sex: missing/noisy portion of data=',len(df[df['sex'].isnull()]) / ent)
    print('capital gain: missing/noisy portion of data=', len(df[df['capital-gain'].isnull() | df['capital-gain']<1]) / ent)
    print('capital loss: missing/noisy portion of data=', len(df[df['capital-loss'].isnull() | df['capital-loss']<1]) / ent)
    print('working hours: missing/noisy portion of data=',len(df[df['hours-per-week'].isnull() | df['hours-per-week']<1]) / ent)
    print('native country: missing/noisy portion of data=',len(df[df['native-country'].isnull()]) / ent)
    print('income: missing/noisy portion of data=',len(df[df['income'].isnull()]) / ent)

df=df.drop(columns=['capital-gain', 'capital-loss']) #drop based on missing data
df=df.drop(columns=['education-num', 'relationship']) # drop based on duplications

#plot the features of integer type
df.boxplot(column=['age']) ;
plt.show()
df.boxplot(column=['fnlwgt']) ;
plt.show()
df.boxplot(column=['hours-per-week']) ;
plt.show()

#plot the features of Categorical type
sns.set(font_scale=0.8)
ax = sns.countplot(y="workclass" , data=df)
plt.show()
ax = sns.countplot(y="education" , data=df)
plt.show()
ax = sns.countplot(y="marital-status" , data=df)
plt.show()
ax = sns.countplot(y="occupation" , data=df)
plt.show()
ax = sns.countplot(y="race" , data=df)
plt.show()
ax = sns.countplot(y="sex" , data=df)
plt.show()
ax = sns.countplot(y="native-country" , data=df)
plt.show()
ax = sns.countplot(y="income" , data=df)
plt.show()


