# model_training.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


def train_models(data):

    features = data.drop(['Target_reg','Target_class'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X_pca, data['Target_reg'], test_size=0.2, shuffle=False)

    _, _, y_train_clf, y_test_clf = train_test_split(
        X_pca, data['Target_class'], test_size=0.2, shuffle=False)

    # Regression
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train_reg)
    reg_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))

    # Classification
    clf = LogisticRegression()
    clf.fit(X_train, y_train_clf)
    clf_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_clf, clf_pred)

    # Clustering
    kmeans = KMeans(n_clusters=3)
    regimes = kmeans.fit_predict(X_pca)

    return rmse, acc, regimes, reg_pred, y_test_reg