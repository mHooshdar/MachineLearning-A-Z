import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# importing the data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# preprocessing
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
column_transformer = ColumnTransformer(transformers=[('a', OneHotEncoder(dtype=int), [0])], remainder='passthrough')
X = column_transformer.fit_transform(X)
# label_encoder_X = LabelEncoder()
# X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
# one_hot_encoder_X = OneHotEncoder(categorical_features=[0], dtype=int)
# X = one_hot_encoder_X.fit_transform(X).toarray()
label_encoder_Y = LabelEncoder()
y = label_encoder_Y.fit_transform(y)

# split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
