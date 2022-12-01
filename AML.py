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
from sklearn.metrics import roc_auc_score


df = pd.read_excel(r"C:\Users\tomls\OneDrive\桌面\AML\AML and ATF Modelling Assignment data_std.xlsx")

y = df['rating']

df = df.drop(columns=['rating','cust_id_masked'])

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

sc = StandardScaler()


import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

X = df
feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(n_estimators=1000,random_state=0)
forest.fit(X_train, y_train)
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
fi_df = pd.DataFrame(importances,index=df.columns)
fi_df=fi_df.sort_values(by=0,ascending=False)
fi_df = fi_df[:20]
#selected_features = fi_df.index.to_numpy()


print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

selected_features = ['cust_tenure', 'prod_total_balance', 'in_person_visit_cnt',
       'avg_cash_5m', 'txn_cash_in_avg_5m', 'prod_asset_balance',
       'avg_wire_12m', 'txn_cash_in_val_5m', 'tot_cash_val_5m',
       'prod_primary_crc_cnt', 'txn_cash_out_avg_5m',
       'prod_primary_cda_cnt', 'prod_prim_acct_cnt']

X = df
feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(n_estimators=1000,random_state=0)

X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.2)
forest.fit(X_train, y_train)

sc = StandardScaler()

y_pred = forest.predict(X_test[selected_features])
sns.heatmap(df[selected_features].corr())
print(y_pred)



y_train = y_train.replace(0,"False")
y_train = y_train.replace(1,"True")
y_test = y_test.replace(0,"False")
y_test = y_test.replace(1,"True")


df_train = X_train
df_train["rating"] = y_train

df_test = X_test
df_test["rating"] = y_test

h2o.init()
# convert pandas DataFrame into H2O Frame
train_h2o = h2o.H2OFrame(df_train)
# Describe  the train h20Frame
train_h2o.describe()

# load test dataset
test_h2o = h2o.H2OFrame(df_test)
# selecting feature and label columns
 
x = test_h2o.columns
x.remove("rating")
y = 'rating'

# Describe  the train h20Frame
test_h2o.describe()

# callh20automl  function
aml = H2OAutoML(max_runtime_secs = 300,
                # max_models =25
                # exclude_algos =['DeepLearning'],
                seed = 1,
                # stopping_metric ='logloss',
                sort_metric ='RMSE',
                balance_classes = False
)
# train model and record time % time
aml.train(x = x, y = y, training_frame = train_h2o)

best_model = aml.get_best_model()
print(best_model)

# Get the metalearner model of top model
metalearner = h2o.get_model(best_model.metalearner()['name'])
 
# list baselearner models :
metalearner.varimp()

predicted_result = best_model.predict(test_h2o)
predicted_result = predicted_result.as_data_frame()
predicted_result.index.name = 'id'
predicted_result.rename(columns = {'predict':'price'}, inplace = True)
predicted_result