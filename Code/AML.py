# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:34:39 2022

@author: tomls
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df = pd.read_excel(r"C:\Users\tomls\OneDrive\桌面\AML\AML and ATF Modelling Assignment data_std.xlsx")

y = df['rating']

df = df.drop(columns=['rating','cust_id_masked'])

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

sc = StandardScaler()

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)


import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

X = df
feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
fi_df = pd.DataFrame(importances,index=df.columns)
fi_df=fi_df.sort_values(by=0,ascending=False)
fi_df = fi_df[:20]
#selected_features = fi_df.index.to_numpy()
sns.heatmap(df[fi_df.index].corr())

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.2)

sc = StandardScaler()

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)