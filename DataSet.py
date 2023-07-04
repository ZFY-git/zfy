import csv

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

csv_name = 'DTXLY_2018.csv'
data = pd.read_csv(f'data/{csv_name}')
sb.pairplot(data.dropna(), hue='label')
plt.savefig(f'picture/{csv_name}_1.png')

# violinplot
plt.figure(figsize=(10,10))
for column_index, column in enumerate(data.columns):
    if column == 'label':
        continue
    plt.subplot(3, 2, column_index + 1)
    sb.violinplot(x='label', y=column, data=data)
    plt.savefig(f'picture/{csv_name}_2.png')

# Obtain feature names
with open(f'data/{csv_name}', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)[:5]
    feature_name = list(header)
    print(feature_name)

# Select the first 4 columns
selected_data = data.iloc[:, :5]
print(selected_data.describe())

# Convert the selected data to a numpy array
selected_data = np.array(selected_data)

# # Reshape the selected data to a dictionary
# selected_data = {'data': selected_data}

selected_target = data.iloc[:,5]

# Convert the selected data to a numpy array
selected_target = np.array(selected_target)
#
# # Reshape the selected data to a dictionary
# selected_target = {'target': selected_target}

X,y = selected_data,selected_target

# 数据分割
# X：待划分的样本数据  y：待划分的样本数据对应的标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, shuffle=None)