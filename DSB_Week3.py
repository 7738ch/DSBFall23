import numpy as np

#question1 
a = np.array([1,2,5,6,8])
b = np.array([1,3,4,7,8])
newV = np.vstack((a,b))
newH = np.hstack((a,b))

#question2
common = np.intersect1d(a,b)

#question3
smallerThanFive = np.where(a<5)

#question4
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
rule = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d[rule]

import pandas as pd

#question1

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
columns = df(pd.DataFrame(columns=list('Manufacturer','Model','Type')))
answer = df[columns.index % 20 == 0]

#question2
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df['Min.Price'] = df['Min.Price'].fillna(df['Min.Price'].mean())
df['Max.Price'] = df['Max.Price'].fillna(df['Max.Price'].mean())

#question3
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
new = df(df.sum(axis=1)>100)

