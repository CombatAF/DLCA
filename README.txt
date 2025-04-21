#1 explore descriptive statistic
import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "C:/Users/ARYA/Downloads/IRIS.csv"
col = 'sepal_length'
df = pd.read_csv(dataset_path)

print('Mean', df[col].mean())
print('Medain', df[col].median())
print('Mode', df[col].mode()[0])
print('Min', df[col].min())
print('Max', df[col].max())
print('Range', df[col].max() - df[col].min())
plt.title('graph')
plt.hist(df[col])
plt.show()

# 2 Apply datacleaning tech

import pandas as pd 

dataset_path = 'C:/Users/ARYA/Downloads/IRIS.csv'
df = pd.read_csv(dataset_path)

df = df.dropna()
df = df.drop_duplicates()
df.columns = df.columns.str.lower()
df['sepal_length'] = df['sepal_length'].astype(float)
df['sepal_petal_length_ratio'] = df['sepal_length'] / df['petal_length']
print(df.head())
print(df['species'].value_counts())


#3 inferencial statistics t test
import pandas as pd
from scipy import stats

dataset_path = 'C:/Users/ARYA/Downloads/IRIS.csv'
df = pd.read_csv(dataset_path)

df = df.dropna()
setosa = df[df['species'] == 'Iris-setosa']['sepal_length']
versicolor = df[df['species'] == 'Iris-versicolor']['sepal_length']

t_stat, p_value = stats.ttest_ind(setosa, versicolor)
print(t_stat)
print(p_value)
if p_value < 0.05:
    print('diff is more')
else:
    print('diff is less')


#4 datavisualization
import pandas as pd
import matplotlib.pyplot as plt 

dataset_path = r"C:\Users\shash\OneDrive\Desktop\ADS prac\IRIS.csv"
df=pd.read_csv(dataset_path)

plt.title('Histogram')
plt.hist(df['sepal_length'])
plt.show()

plt.title('Bar chart')
plt.scatter(df['petal_length'], df['sepal_length'])
plt.show()

plt.title('Pie chart')
plt.pie(df['species'].value_counts(), labels = df['species'].value_counts().index)
plt.show()


#5 linear regression and performance evaluation metrics
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset_path = 'C:/Users/ARYA/Downloads/IRIS.csv'
df = pd.read_csv(dataset_path)

X, Y = df[['sepal_width']], df['sepal_length']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42, test_size = 0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print('MAE', mean_absolute_error(Y_test, Y_pred))
print('MSE', mean_squared_error(Y_test, Y_pred))
print('R2', r2_score(Y_test, Y_pred))

#6 SMOTE
import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

dataset_path = 'C:/Users/ARYA/Downloads/IRIS.csv'
df = pd.read_csv(dataset_path)

X, Y = df[['sepal_width']], df['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42, test_size = 0.2)
print(Y_train.value_counts())

smote = SMOTE(random_state = 42)
X_res, Y_res = smote.fit_resample(X_train, Y_train)
print(Y_res.value_counts())

#7 DBSCAN
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset_path = 'C:/Users/ARYA/Downloads/IRIS.csv'
df = pd.read_csv(dataset_path)

X = df[['sepal_length', 'petal_width']]
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.title('DBSCAN')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c = clusters)
plt.show()

#8 Moving avg
# 8 Time series decomp
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose

dataset_path = 'C:/Users/ARYA/Downloads/IRIS.csv'
df = pd.read_csv(dataset_path)

df['MA'] = df['sepal_length'].rolling(window = 10).mean()
plt.plot(df['sepal_length'], label = 'original')
plt.plot(df['MA'], label = 'Moving Average', color = 'red')
plt.legend()
plt.show()

result = seasonal_decompose(df['sepal_length'], model = 'additive', period = 10)
result.plot()
plt.title('Time Series Decomposition')
plt.show()