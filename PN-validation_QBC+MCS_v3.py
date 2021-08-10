import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling
from datetime import datetime

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

batchSize = 20             # size of the training set increments
seedSize = 10              # size of the initial training set
num_iters_test = 5       # choose desired number of test loop iterations
size_test_set = 0.1        # choose desired size of the test set
numCommitteeMembers = 5    # choose number of committee members for the QBC


# -------->  CREATION OF LISTS CONTAINING THE PREDICTIONS AND TARGETS  <------------------
preds_rr_mcs, preds_rr_qbc = [], []
preds_mlp_mcs, preds_mlp_qbc = [], []
preds_xgb_mcs, preds_xgb_qbc = [], []
targets_rr_mcs, targets_rr_qbc = [], []
targets_mlp_mcs, targets_mlp_qbc = [], []
targets_xgb_mcs, targets_xgb_qbc = [], []
targets_rr_qbc_mlp, preds_rr_qbc_mlp = [], []
targets_mlp_qbc_mlp, preds_mlp_qbc_mlp = [], []
targets_xgb_qbc_mlp, preds_xgb_qbc_mlp = [], []
targets_rr_qbc_3, preds_rr_qbc_3 = [], []
targets_mlp_qbc_3, preds_mlp_qbc_3 = [], []
targets_xgb_qbc_3, preds_xgb_qbc_3 = [], []


# -------->  FUNCTIONS  <------------------
# RMSE CALCULATION
def rmse(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2.
    return np.sqrt(residuals / n)


def normalize(input_array):
    mean = np.mean(input_array, axis=0)
    std = np.std(input_array, axis=0)

    # scikit-learn measure to handle zeros in scale: def _handle_zeros_in_scale(scale, copy=True)
    # https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70
    if np.isscalar(std):
        if std == .0:
            std = 1.
    elif isinstance(std, np.ndarray):
        std = std.copy()
        std[std == 0.0] = 1.0

    data_norm = (input_array - mean) / std
    return mean, std, data_norm


# -------->  MACHINE LEARNING MODELS (HYPERPARAMETERS, COMMITTEE FOR QBC) <------------------

# Ridge Regression RR
rr_params_v1 = {
    'alpha': 5, 'max_iter': 4, 'normalize': False, 'solver': 'lsqr', 'tol': 0.003}
rr_params_v2 = {
    'alpha': 6, 'max_iter': 8, 'normalize': False, 'solver': 'lsqr', 'tol': 0.002}
rr_params_v3 = {
    'alpha': 7, 'max_iter': 16, 'normalize': False, 'solver': 'lsqr', 'tol': 0.004}
rr_params_v4 = {
    'alpha': 5, 'max_iter': 16, 'normalize': False, 'solver': 'lsqr', 'tol': 0.002}
rr_params_v5 = {
    'alpha': 7, 'max_iter': 32, 'normalize': False, 'solver': 'lsqr', 'tol': 0.008}

mcs_rr_regr = Ridge(**rr_params_v1)
qbc_rr_regr = Ridge(**rr_params_v1)


# Multilayer Perceptron Neural Network MLP
mlp_params_v1 = {
    'solver': 'adam', 'hidden_layer_sizes': (60, 60), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 3000}
mlp_params_v2 = {
    'solver': 'adam', 'hidden_layer_sizes': (55, 55), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 3000}
mlp_params_v3 = {
    'solver': 'adam', 'hidden_layer_sizes': (65, 65), 'activation': 'relu', 'tol': 1e-4, 'max_iter': 3000}
mlp_params_v4 = {
    'solver': 'adam', 'hidden_layer_sizes': (30, 30), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 3000}
mlp_params_v5 = {
    'solver': 'adam', 'hidden_layer_sizes': (20, 60, 20), 'activation': 'relu', 'tol': 1e-4, 'max_iter': 3000}

mlp_params_list = [mlp_params_v1, mlp_params_v3, mlp_params_v4]
mcs_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_mlp_regr = MLPRegressor(**mlp_params_v1)


# eXtreme Gradient Boosting XGB
xgb_params_v1 = {
    'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
    'booster': 'gbtree', 'n_jobs': -1}

mcs_xgb_regr = XGBRegressor(**xgb_params_v1)
qbc_xgb_regr = XGBRegressor(**xgb_params_v1)

_3_estimators_list = [Ridge(**rr_params_v1), MLPRegressor(**mlp_params_v1), XGBRegressor(**xgb_params_v1)]
# initializing Committee members
num_committee_members = 5

# -------->  DATA INPUT  <------------------
df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/MLPlatform/data/auto_mpg.csv', header=0)
df_data = df_data.fillna(value=0)
arr_data = np.asarray(df_data)
trainingSetSizes = np.arange(seedSize, arr_data.shape[0] * (1-size_test_set), batchSize)


# -------->  MONTE CARLO SAMPLING ACTIVE LEARNING (MCS) DOEs  <------------------
print('calculating MCS...')

# -------->  ITERATIONS OVER DIFFERENT TEST SETS: TEST LOOP  <------------------

for test_loop in range(num_iters_test):
    startTime = datetime.now()

    # -------->  PREPARATION TEST SET <------------------
    np.random.seed(test_loop)
    pool = arr_data.copy()
    np.random.shuffle(pool)
    idx_split = int(pool.shape[0] * (1-size_test_set))
    X_train = pool[:idx_split, :-1]
    X_test = pool[idx_split:, :-1]
    y_train = pool[:idx_split, -1].reshape((-1, 1))
    y_test = pool[idx_split:, -1].reshape((-1, 1))


    for lc_loop in range(len(trainingSetSizes)):
        startTime_batch = datetime.now()
        print(str(test_loop) + '.' + str(lc_loop))

        # -------->  MONTE CARLO SAMPLING (MCS) <------------------
        mcs_X_train_split = X_train[:int(trainingSetSizes[lc_loop]), :]
        mcs_y_train_split = y_train[:int(trainingSetSizes[lc_loop])]

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(mcs_X_train_split)
        mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(mcs_y_train_split)

        # -------->  MODEL FITTING  <------------------
        mcs_mlp_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
        mcs_xgb_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
        mcs_rr_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())

        # -------->  PREDICTION <------------------
        mcs_mlp_y_norm = mcs_mlp_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
        mcs_xgb_y_norm = mcs_xgb_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
        mcs_rr_y_norm = mcs_rr_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)

        # -------->  DENORMALIZATION OF PREDICTED VALUES <------------------
        mcs_mlp_y = mcs_mlp_y_norm * mcs_std_y_train + mcs_mean_y_train
        mcs_xgb_y = mcs_xgb_y_norm * mcs_std_y_train + mcs_mean_y_train
        mcs_rr_y = mcs_rr_y_norm * mcs_std_y_train + mcs_mean_y_train

        # -------->  SAVING PREDICTED VALUES <------------------
        preds_rr_mcs.append(mcs_rr_y)
        preds_mlp_mcs.append(mcs_mlp_y)
        preds_xgb_mcs.append(mcs_xgb_y)
        targets_rr_mcs.append(y_test)
        targets_mlp_mcs.append(y_test)
        targets_xgb_mcs.append(y_test)

        print('   ' + str(datetime.now() - startTime_batch))
    print('Test loop run time: ' + str(datetime.now() - startTime))




# -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
print('calculating QBC...')

for test_loop in range(num_iters_test):
    startTime = datetime.now()

    # -------->  PREPARATION TEST SET <------------------
    np.random.seed(test_loop)
    pool = arr_data.copy()
    np.random.shuffle(pool)
    idx_split = int(pool.shape[0] * (1 - size_test_set))
    X_train = pool[:idx_split, :-1]
    X_test = pool[idx_split:, :-1]
    y_train = pool[:idx_split, -1].reshape((-1, 1))
    y_test = pool[idx_split:, -1].reshape((-1, 1))

    qbc_X_train_split = X_train[:seedSize, :]
    qbc_y_train_split = y_train[:seedSize, :].reshape((-1, 1))

    qbc_X_test_split = X_train[seedSize:, :]
    qbc_y_test_split = y_train[seedSize:, :].reshape((-1, 1))

    for idx, batch in enumerate(trainingSetSizes):
        startTime_batch = datetime.now()
        print(str(test_loop) + '.' + str(idx))

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)
        qbc_X_test_split_norm = (qbc_X_test_split - qbc_mean_X_train) / qbc_std_X_train
        qbc_y_test_split_norm = (qbc_y_test_split - qbc_mean_y_train) / qbc_std_y_train

        X_test_norm = (X_test - qbc_mean_X_train) / qbc_std_X_train
        y_test_norm = (y_test - qbc_mean_y_train) / qbc_std_y_train

        # -------->  MODEL FITTING  <------------------
        qbc_mlp_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_xgb_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_rr_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())

        # -------->  PREDICTION  <------------------
        qbc_rr_y_norm = qbc_rr_regr.predict(X_test_norm)
        qbc_mlp_y_norm = qbc_mlp_regr.predict(X_test_norm)
        qbc_xgb_y_norm = qbc_xgb_regr.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        qbc_mlp_y = qbc_mlp_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_xgb_y = qbc_xgb_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_rr_y = qbc_rr_y_norm * qbc_std_y_train + qbc_mean_y_train

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_rr_qbc.append(qbc_rr_y)
        preds_mlp_qbc.append(qbc_mlp_y)
        preds_xgb_qbc.append(qbc_xgb_y)
        targets_rr_qbc.append(y_test)
        targets_mlp_qbc.append(y_test)
        targets_xgb_qbc.append(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break

        else:
            # --------> CREATION OF THE COMMITTEE) <------------------
            learner_list = []
            for i in range(numCommitteeMembers):
                learner_list.append(ActiveLearner(estimator=Ridge(**rr_params_v1),
                                                  X_training=qbc_X_train_split_norm,
                                                  y_training=qbc_y_train_split_norm.ravel(),
                                                  bootstrap_init=True))

            committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

            # -------->  PREPARING NEXT BATCH <------------------
            for query in range(batchSize):
                # sample selection and committee training
                query_idx, query_instance = committee.query(qbc_X_test_split_norm)
                committee.teach(qbc_X_test_split_norm[query_idx], qbc_y_test_split_norm[query_idx], bootstrap=True)
                committee.rebag()

                qbc_X_train_split = np.append(qbc_X_train_split, qbc_X_test_split[query_idx], axis=0)
                qbc_y_train_split = np.append(qbc_y_train_split, qbc_y_test_split[query_idx], axis=0)
                qbc_X_train_split_norm = np.append(qbc_X_train_split_norm, qbc_X_test_split_norm[query_idx], axis=0)
                qbc_y_train_split_norm = np.append(qbc_y_train_split_norm, qbc_y_test_split_norm[query_idx], axis=0)

                qbc_X_test_split = np.delete(qbc_X_test_split, query_idx, axis=0)
                qbc_y_test_split = np.delete(qbc_y_test_split, query_idx, axis=0)
                qbc_X_test_split_norm = np.delete(qbc_X_test_split_norm, query_idx, axis=0)
                qbc_y_test_split_norm = np.delete(qbc_y_test_split_norm, query_idx, axis=0)

        print('   ' + str(datetime.now() - startTime_batch))
    print('Test loop run time: ' + str(datetime.now() - startTime))


# -------->  MLP QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
print('calculating QBC_MLP...')

for test_loop in range(num_iters_test):
    startTime = datetime.now()

    # -------->  PREPARATION TEST SET <------------------
    np.random.seed(test_loop)
    pool = arr_data.copy()
    np.random.shuffle(pool)
    idx_split = int(pool.shape[0] * (1 - size_test_set))
    X_train = pool[:idx_split, :-1]
    X_test = pool[idx_split:, :-1]
    y_train = pool[:idx_split, -1].reshape((-1, 1))
    y_test = pool[idx_split:, -1].reshape((-1, 1))

    qbc_X_train_split = X_train[:seedSize, :]
    qbc_y_train_split = y_train[:seedSize, :].reshape((-1, 1))

    qbc_X_test_split = X_train[seedSize:, :]
    qbc_y_test_split = y_train[seedSize:, :].reshape((-1, 1))

    for idx, batch in enumerate(trainingSetSizes):
        startTime_batch = datetime.now()
        print(str(test_loop) + '.' + str(idx))

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)
        qbc_X_test_split_norm = (qbc_X_test_split - qbc_mean_X_train) / qbc_std_X_train
        qbc_y_test_split_norm = (qbc_y_test_split - qbc_mean_y_train) / qbc_std_y_train

        X_test_norm = (X_test - qbc_mean_X_train) / qbc_std_X_train
        y_test_norm = (y_test - qbc_mean_y_train) / qbc_std_y_train

        # -------->  MODEL FITTING  <------------------
        qbc_mlp_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_xgb_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_rr_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())

        # -------->  PREDICTION  <------------------
        qbc_rr_y_norm = qbc_rr_regr.predict(X_test_norm)
        qbc_mlp_y_norm = qbc_mlp_regr.predict(X_test_norm)
        qbc_xgb_y_norm = qbc_xgb_regr.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        qbc_mlp_y = qbc_mlp_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_xgb_y = qbc_xgb_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_rr_y = qbc_rr_y_norm * qbc_std_y_train + qbc_mean_y_train

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_rr_qbc_mlp.append(qbc_rr_y)
        preds_mlp_qbc_mlp.append(qbc_mlp_y)
        preds_xgb_qbc_mlp.append(qbc_xgb_y)
        targets_rr_qbc_mlp.append(y_test)
        targets_mlp_qbc_mlp.append(y_test)
        targets_xgb_qbc_mlp.append(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break

        else:
            # --------> CREATION OF THE COMMITTEE) <------------------
            learner_list = []
            for params in mlp_params_list:
                learner_list.append(ActiveLearner(estimator=MLPRegressor(**params),
                                                  X_training=qbc_X_train_split_norm,
                                                  y_training=qbc_y_train_split_norm.ravel(),
                                                  bootstrap_init=True))

            committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

            # -------->  PREPARING NEXT BATCH <------------------
            for query in range(batchSize):
                # sample selection and committee training
                query_idx, query_instance = committee.query(qbc_X_test_split_norm)
                committee.teach(qbc_X_test_split_norm[query_idx], qbc_y_test_split_norm[query_idx].ravel(), bootstrap=True)
                committee.rebag()

                qbc_X_train_split = np.append(qbc_X_train_split, qbc_X_test_split[query_idx], axis=0)
                qbc_y_train_split = np.append(qbc_y_train_split, qbc_y_test_split[query_idx], axis=0)
                qbc_X_train_split_norm = np.append(qbc_X_train_split_norm, qbc_X_test_split_norm[query_idx], axis=0)
                qbc_y_train_split_norm = np.append(qbc_y_train_split_norm, qbc_y_test_split_norm[query_idx], axis=0)

                qbc_X_test_split = np.delete(qbc_X_test_split, query_idx, axis=0)
                qbc_y_test_split = np.delete(qbc_y_test_split, query_idx, axis=0)
                qbc_X_test_split_norm = np.delete(qbc_X_test_split_norm, query_idx, axis=0)
                qbc_y_test_split_norm = np.delete(qbc_y_test_split_norm, query_idx, axis=0)

        print('   ' + str(datetime.now() - startTime_batch))
    print('Test loop run time: ' + str(datetime.now() - startTime))


# -------->  3-QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
print('calculating QBC_3...')

for test_loop in range(num_iters_test):
    startTime = datetime.now()

    # -------->  PREPARATION TEST SET <------------------
    np.random.seed(test_loop)
    pool = arr_data.copy()
    np.random.shuffle(pool)
    idx_split = int(pool.shape[0] * (1 - size_test_set))
    X_train = pool[:idx_split, :-1]
    X_test = pool[idx_split:, :-1]
    y_train = pool[:idx_split, -1].reshape((-1, 1))
    y_test = pool[idx_split:, -1].reshape((-1, 1))

    qbc_X_train_split = X_train[:seedSize, :]
    qbc_y_train_split = y_train[:seedSize, :].reshape((-1, 1))

    qbc_X_test_split = X_train[seedSize:, :]
    qbc_y_test_split = y_train[seedSize:, :].reshape((-1, 1))

    for idx, batch in enumerate(trainingSetSizes):
        startTime_batch = datetime.now()
        print(str(test_loop) + '.' + str(idx))

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)
        qbc_X_test_split_norm = (qbc_X_test_split - qbc_mean_X_train) / qbc_std_X_train
        qbc_y_test_split_norm = (qbc_y_test_split - qbc_mean_y_train) / qbc_std_y_train

        X_test_norm = (X_test - qbc_mean_X_train) / qbc_std_X_train
        y_test_norm = (y_test - qbc_mean_y_train) / qbc_std_y_train

        # -------->  MODEL FITTING  <------------------
        qbc_mlp_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_xgb_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_rr_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())

        # -------->  PREDICTION  <------------------
        qbc_rr_y_norm = qbc_rr_regr.predict(X_test_norm)
        qbc_mlp_y_norm = qbc_mlp_regr.predict(X_test_norm)
        qbc_xgb_y_norm = qbc_xgb_regr.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        qbc_mlp_y = qbc_mlp_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_xgb_y = qbc_xgb_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_rr_y = qbc_rr_y_norm * qbc_std_y_train + qbc_mean_y_train

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_rr_qbc_3.append(qbc_rr_y)
        preds_mlp_qbc_3.append(qbc_mlp_y)
        preds_xgb_qbc.append(qbc_xgb_y)
        targets_rr_qbc.append(y_test)
        targets_mlp_qbc.append(y_test)
        targets_xgb_qbc.append(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break

        else:
            # --------> CREATION OF THE COMMITTEE) <------------------
            learner_list = []
            for params in _3_estimators_list:
                learner_list.append(ActiveLearner(estimator=params,
                                                  X_training=qbc_X_train_split_norm,
                                                  y_training=qbc_y_train_split_norm.ravel(),
                                                  bootstrap_init=True))

            committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

            # -------->  PREPARING NEXT BATCH <------------------
            for query in range(batchSize):
                # sample selection and committee training
                query_idx, query_instance = committee.query(qbc_X_test_split_norm)
                committee.teach(qbc_X_test_split_norm[query_idx], qbc_y_test_split_norm[query_idx].ravel(), bootstrap=True)
                committee.rebag()

                qbc_X_train_split = np.append(qbc_X_train_split, qbc_X_test_split[query_idx], axis=0)
                qbc_y_train_split = np.append(qbc_y_train_split, qbc_y_test_split[query_idx], axis=0)
                qbc_X_train_split_norm = np.append(qbc_X_train_split_norm, qbc_X_test_split_norm[query_idx], axis=0)
                qbc_y_train_split_norm = np.append(qbc_y_train_split_norm, qbc_y_test_split_norm[query_idx], axis=0)

                qbc_X_test_split = np.delete(qbc_X_test_split, query_idx, axis=0)
                qbc_y_test_split = np.delete(qbc_y_test_split, query_idx, axis=0)
                qbc_X_test_split_norm = np.delete(qbc_X_test_split_norm, query_idx, axis=0)
                qbc_y_test_split_norm = np.delete(qbc_y_test_split_norm, query_idx, axis=0)

        print('   ' + str(datetime.now() - startTime_batch))
    print('Test loop run time: ' + str(datetime.now() - startTime))

# -------->  SAVING THE RESULTS  <------------------
print("Saving...")

np.savetxt("preds_rr_mcs.csv", np.ravel(preds_rr_mcs), delimiter=",")
np.savetxt("preds_mlp_mcs.csv", np.ravel(preds_mlp_mcs), delimiter=",")
np.savetxt("preds_xgb_mcs.csv", np.ravel(preds_xgb_mcs), delimiter=",")

np.savetxt("targets_rr_mcs.csv", np.ravel(targets_rr_mcs), delimiter=",")
np.savetxt("targets_mlp_mcs.csv", np.ravel(targets_mlp_mcs), delimiter=",")
np.savetxt("targets_xgb_mcs.csv", np.ravel(targets_xgb_mcs), delimiter=",")

np.savetxt("preds_rr_qbc.csv", np.ravel(preds_rr_qbc), delimiter=",")
np.savetxt("preds_mlp_qbc.csv", np.ravel(preds_mlp_qbc), delimiter=",")
np.savetxt("preds_xgb_qbc.csv", np.ravel(preds_xgb_qbc), delimiter=",")

np.savetxt("targets_rr_qbc.csv", np.ravel(targets_rr_qbc), delimiter=",")
np.savetxt("targets_mlp_qbc.csv", np.ravel(targets_mlp_qbc), delimiter=",")
np.savetxt("targets_xgb_qbc.csv", np.ravel(targets_xgb_qbc), delimiter=",")

np.savetxt("preds_rr_qbc_mlp.csv", np.ravel(preds_rr_qbc_mlp), delimiter=",")
np.savetxt("preds_mlp_qbc_mlp.csv", np.ravel(preds_mlp_qbc_mlp), delimiter=",")
np.savetxt("preds_xgb_qbc_mlp.csv", np.ravel(preds_xgb_qbc_mlp), delimiter=",")

np.savetxt("targets_rr_qbc_mlp.csv", np.ravel(targets_rr_qbc_mlp), delimiter=",")
np.savetxt("targets_mlp_qbc_mlp.csv", np.ravel(targets_mlp_qbc_mlp), delimiter=",")
np.savetxt("targets_xgb_qbc_mlp.csv", np.ravel(targets_xgb_qbc_mlp), delimiter=",")

np.savetxt("preds_rr_qbc_3.csv", np.ravel(preds_rr_qbc_3), delimiter=",")
np.savetxt("preds_mlp_qbc_3.csv", np.ravel(preds_mlp_qbc_3), delimiter=",")
np.savetxt("preds_xgb_qbc_3.csv", np.ravel(preds_xgb_qbc_3), delimiter=",")

np.savetxt("targets_rr_qbc_3.csv", np.ravel(targets_rr_qbc_3), delimiter=",")
np.savetxt("targets_mlp_qbc_3.csv", np.ravel(targets_mlp_qbc_3), delimiter=",")
np.savetxt("targets_xgb_qbc_3.csv", np.ravel(targets_xgb_qbc_3), delimiter=",")

print("Done.")
