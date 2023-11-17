import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

glass = pd.read_csv('glass.csv')

#Question 1 

glass['binary_target'] = glass['Type'].apply(lambda x: 1 if x == 1 else 0)

X = glass.drop(['Type', 'binary_target'], axis=1)
y = glass['binary_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0, 1, 101)
accuracy_scores = []
precision_scores = []
recall_scores = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)

results_df = pd.DataFrame({
    'Threshold': thresholds,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores
})

#Question 2

unique_types = glass['Type'].unique()

type_results = {}

for glass_type in unique_types:
    glass['binary_target'] = glass['Type'].apply(lambda x: 1 if x == glass_type else 0)
    X = glass.drop(['Type', 'binary_target'], axis=1)
    y = glass['binary_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

    results_df = pd.DataFrame({
        'Threshold': thresholds,
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores
    })

    type_results[glass_type] = results_df

#Question 3

X = glass.drop('Type', axis=1)
y = glass['Type']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

#Question 4

X = glass.drop('Type', axis=1)
y = glass['Type']

y_binarized = label_binarize(y, classes=np.unique(y))

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_binarized, test_size=0.3, random_state=42)

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)

n_classes = y_binarized.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], model.decision_function(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'brown', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for each class')
plt.legend(loc="lower right")
plt.show()

#Question 5

%matplotlib inline 
import seaborn as sns
plt.style.use('fivethirtyeight')

df = pd.read_csv("iris.csv")
print(df['Name'].value_counts())
df.head(5)

cols = df.columns[:-1]
sns.pairplot(df[cols])

X_scaled = preprocessing.MinMaxScaler().fit_transform(df[cols])
pd.DataFrame(X_scaled, columns=cols).describe()

k = 5
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

metrics.silhouette_score(X_scaled, labels, metric='euclidean')

df['label'] = labels
df.head()

cols = df.columns[:-2]
sns.pairplot(df, x_vars=cols, y_vars= cols, hue='label')

sns.pairplot(df, x_vars=cols, y_vars= cols, hue='Name')

#Question 6

nutrients = pd.read_csv("nutrients.txt")
print(nutrients['Name'].value_counts())
nutrients.head(5)

cols = nutrients.columns[:-1]
sns.pairplot(nutrients[cols])

X_scaled = preprocessing.MinMaxScaler().fit_transform(nutrients[cols])
pd.DataFrame(X_scaled, columns=cols).describe()

k = 2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

metrics.silhouette_score(X_scaled, labels, metric='euclidean')

nutrients['label'] = labels
nutrients.head()

cols = nutrients.columns[:-2]
sns.pairplot(nutrients, x_vars=cols, y_vars= cols, hue='label')

sns.pairplot(nutrients, x_vars=cols, y_vars= cols, hue='Name')