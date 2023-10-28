import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#question1
data = pd.read_csv('boston_housing_data.csv')

plt.style.use('classic')  
plt.figure(figsize=(12, 8))
plt.plot(data['ZN'], color='green', linestyle='-', linewidth=2, label='ZN')
plt.plot(data['INDUS'], color='blue', linestyle='--', linewidth=2, label='INDUS')
plt.title('Line Plot of ZN and INDUS')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

#question2

#vertical
dataDummy = pd.DataFrame({
    'col1': [10, 20, 30, 40, 50],
    'col2': [20, 10, 40, 30, 50]
})

fig, ax = plt.subplots()
ax.bar(dataDummy.index, dataDummy['col1'], label='col1')
ax.bar(dataDummy.index, dataDummy['col2'], bottom=dataDummy['col1'], label='col2')
ax.set_title('Large Title: Vertical Bar Chart')
ax.legend(loc='lower left')
plt.show()

#horizontal
fig, ax = plt.subplots()
ax.barh(data.index, data['col1'], label='col1')
ax.barh(data.index, data['col2'], left=data['col1'], label='col2')
ax.set_title('Large Title: Horizontal Bar Chart')
ax.legend(loc='upper right')
plt.show()

#question3

plt.figure(figsize=(12, 8))  
data['MEDV'].hist(bins=20, edgecolor='k')
plt.title('Histogram of MEDV')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()

#question4

plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='X', y='Y')
plt.title('Scatter Plot of X vs Y')
plt.show()

#question5

corr_matrix = data.corr()
sns.scatterplot(data=data, x='X', y='Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y')
plt.show()