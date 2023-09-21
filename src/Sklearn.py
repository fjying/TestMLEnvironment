#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def main():
    start_time = time.time()

    # Directory path
    directory_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print(directory_path)

    # Change to Data Path
    os.chdir(os.path.join(directory_path, 'data'))

    # LabelEncoder
    # Load Iris Dataset
    iris_df = pd.read_csv('iris.csv')
    X = iris_df.iloc[:, :-1].to_numpy()
    le_plant = LabelEncoder()
    le_plant.fit(iris_df['class'])
    y = le_plant.transform(iris_df.iloc[:, -1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =1 , stratify=y)

    # Standardizing the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Training a perceptron via scikit-learn
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    total_run_time = time.time()-start_time
    print('Total Run Time', total_run_time)
if __name__ == '__main__':
    main()