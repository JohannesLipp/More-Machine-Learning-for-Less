import numpy as np
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling
from sklearn.model_selection import train_test_split
import pandas as pd


df_data = pd.read_csv(
                '/Users/philippnoodt/Studium/RWTH/B.Sc. Maschinenbau Energietechnik/BA/Python_BA/ba-philipp-code/'
                'Simulated Data/data_preprocessed_DOC2.csv', header=0)

data = np.asarray(df_data)
data_norm = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)

# MLP Regressor
mlp_solver, mlp_hidden_layer_sizes, mlp_activation = 'adam', (60, 60), 'relu'
mlp_tol, mlp_maxiter = 1e-5, 300
mlp_regr = MLPRegressor(solver=mlp_solver, hidden_layer_sizes=mlp_hidden_layer_sizes,
                            activation=mlp_activation, tol=mlp_tol, max_iter=mlp_maxiter)
# Ridge Regression RR
rr_alpha, rr_max_iter, rr_normalize = 5, 4, False
rr_solver, rr_tol = 'lsqr', 0.003
rr_regr = Ridge(alpha=rr_alpha, max_iter=rr_max_iter, normalize=rr_normalize,
                    solver=rr_solver, tol=rr_tol)
# eXtreme Gradient Boosting XGB
xgb_max_depth, xgb_learning_rate, xgb_n_estimaters, xgb_eta = 2, 0.3, 1250, 0.3
min_child_weight, booster = 5, 'gbtree'
xgb_regr = XGBRegressor(max_depth=xgb_max_depth, learning_rate=xgb_learning_rate, booster=booster,
                            n_estimators=xgb_n_estimaters, silent=1, eta=xgb_eta, min_child_weight=min_child_weight)

# RMSE CALCULATION
def rmse(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2.
    return np.sqrt(residuals / n)

def ensemble_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

# create the data pool
X_pool, X_test, y_pool, y_test = train_test_split(data_norm[:, :-1], data_norm[:, -1], test_size=0.15)
print(type(X_pool), X_pool.shape, type(y_pool), y_pool.shape)
X_pool_copy = X_pool.copy()

# initializing Committee members
n_members = 3
learner_list = list()

# initial training data
n_initial = 10
train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
X_train = X_pool[train_idx]
y_train = y_pool[train_idx]
X_pool = np.delete(X_pool, train_idx, axis=0)
y_pool = np.delete(y_pool, train_idx)

#print(y_pool[train_idx])
print(type(X_pool), X_pool.shape, type(y_pool), y_pool.shape)

print(np.max(y_pool), np.min(y_pool), np.max(y_test), np.min(y_test))

for member_idx in range(n_members):
    # initializing learner
    learner = ActiveLearner(
        # estimator=XGBRegressor(),
        estimator=MLPRegressor(solver='adam', hidden_layer_sizes=(60, 60), activation='relu', tol=1e-5, max_iter=300),
        #RandomForestRegressor(, MLPRegressor(
        X_training=X_train, y_training=y_train.ravel()
        )
    learner_list.append(learner)

learner_list_1 = []
learner_list_1.append(ActiveLearner(
    estimator=MLPRegressor(solver='adam', hidden_layer_sizes=(59, 59), activation='relu', tol=1e-5,
                       max_iter=300),
    X_training=X_train, y_training=y_train.ravel()
))
learner_list_1.append(ActiveLearner(
    estimator=MLPRegressor(solver='adam', hidden_layer_sizes=(60, 60), activation='relu', tol=1e-5,
                       max_iter=400),
    X_training=X_train, y_training=y_train.ravel()
))
learner_list_1.append(ActiveLearner(
    estimator=MLPRegressor(solver='adam', hidden_layer_sizes=(61, 61), activation='relu', tol=1e-5,
                       max_iter=500),  # RandomForestRegressor(, MLPRegressor(
    X_training=X_train, y_training=y_train.ravel()
))
2, 0.3, 1250, 0.3
learner_list_2 = []
learner_list_2.append(ActiveLearner(
    estimator=XGBRegressor(max_depth=2, learning_rate=0.29, booster='gbtree',
                            n_estimators=1200, silent=1, eta=0.29, min_child_weight=5),
    X_training=X_train, y_training=y_train.ravel()
))
learner_list_2.append(ActiveLearner(
    estimator=XGBRegressor(max_depth=2, learning_rate=0.3, booster='gbtree',
                            n_estimators=1250, silent=1, eta=0.3, min_child_weight=5),
    X_training=X_train, y_training=y_train.ravel()
))
learner_list_2.append(ActiveLearner(
    estimator=XGBRegressor(max_depth=2, learning_rate=0.31, booster='gbtree',
                            n_estimators=1300, silent=1, eta=0.31, min_child_weight=5),
    X_training=X_train, y_training=y_train.ravel()
))


committee = CommitteeRegressor(learner_list=learner_list_2, query_strategy=max_std_sampling)

n_queries = 100
train_mlp = X_train
#train_mlp.append(X_train)
target_mlp = y_train
#target_mlp.append(y_train)

for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(X_pool[query_idx], y_pool[query_idx])
    train_mlp = np.concatenate((train_mlp, X_pool[query_idx]), axis=0)
    target_mlp = np.concatenate((target_mlp, y_pool[query_idx]), axis=0)
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
    predictions_norm, std = committee.predict(X_test, return_std=True)
    predictions = predictions_norm * data_std[-1] + data_mean[-1]
    rmse_result = rmse((y_test * data_std[-1] + data_mean[-1]), predictions)
    print(idx, np.round(rmse_result * 1000, decimals=2), np.round(np.mean(std), decimals=5), query_instance, query_idx)
    #print('stop')


print(type(X_pool), X_pool.shape, type(y_pool), y_pool.shape)
print('stop')
print(np.shape(train_mlp))
train_mlp_std = (train_mlp - np.mean(train_mlp)) / np.std(train_mlp)
target_mlp_std = (target_mlp - np.mean(target_mlp)) / np.std(target_mlp)
mlp_regr.fit(X=train_mlp_std, y=target_mlp_std)
preds = mlp_regr.predict(X=(X_test -np.mean(X_test)) / np.std(X_test))
rmse_result_mlp = rmse(y_test, (preds * np.std(target_mlp) + np.mean(target_mlp)))
print(rmse_result_mlp*1000)