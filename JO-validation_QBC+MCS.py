import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling
import random

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

seedSize = 10
batchSize = 10
num_iters_test = 20  # choose desired number of test loop iterations
numCommitteeMembers = 5


# -------->  CREATION OF LISTS CONTAINING THE PREDICTIONS  <------------------
preds_rr_mcs, preds_rr_qbc = [], []
preds_mlp_mcs, preds_mlp_qbc = [], []
preds_xgb_mcs, preds_xgb_qbc = [], []
targets_rr_mcs, targets_rr_qbc = [], []
targets_mlp_mcs, targets_mlp_qbc = [], []
targets_xgb_mcs, targets_xgb_qbc = [], []
preds_rr_mcs = np.asarray(preds_rr_mcs)
preds_mlp_mcs = np.asarray(preds_mlp_mcs)
preds_xgb_mcs = np.asarray(preds_xgb_mcs)
preds_rr_qbc = np.asarray(preds_rr_qbc)
preds_mlp_qbc = np.asarray(preds_mlp_qbc)
preds_xgb_qbc = np.asarray(preds_xgb_qbc)
targets_rr_mcs = np.asarray(targets_rr_mcs)
targets_mlp_mcs = np.asarray(targets_mlp_mcs)
targets_xgb_mcs = np.asarray(targets_xgb_mcs)
targets_rr_qbc = np.asarray(targets_rr_qbc)
targets_mlp_qbc = np.asarray(targets_mlp_qbc)
targets_xgb_qbc = np.asarray(targets_xgb_qbc)


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
lhs_rr_regr = Ridge(**rr_params_v1)
mcs_rr_regr = Ridge(**rr_params_v1)
bbd_rr_regr = Ridge(**rr_params_v1)
ccd_rr_regr = Ridge(**rr_params_v1)
qbc_rr_regr = Ridge(**rr_params_v1)
qbc_commi_rr_regr = Ridge(**rr_params_v1)

# Multilayer Perceptron Neural Network MLP
mlp_params_v1 = {
    'solver': 'adam', 'hidden_layer_sizes': (60, 60), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 300}
mlp_params_v2 = {
    'solver': 'adam', 'hidden_layer_sizes': (55, 55), 'activation': 'relu', 'tol': 1e-5}
mlp_params_v3 = {
    'solver': 'adam', 'hidden_layer_sizes': (65, 65), 'activation': 'relu', 'tol': 1e-4}

lhs_mlp_regr = MLPRegressor(**mlp_params_v1)
mcs_mlp_regr = MLPRegressor(**mlp_params_v1)
bbd_mlp_regr = MLPRegressor(**mlp_params_v1)
ccd_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_commi_mlp_regr = MLPRegressor(**mlp_params_v1)

# eXtreme Gradient Boosting XGB
xgb_params = {
    'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
    'booster': 'gbtree', 'n_jobs': -1}

lhs_xgb_regr = XGBRegressor(**xgb_params)
mcs_xgb_regr = XGBRegressor(**xgb_params)
bbd_xgb_regr = XGBRegressor(**xgb_params)
ccd_xgb_regr = XGBRegressor(**xgb_params)
qbc_xgb_regr = XGBRegressor(**xgb_params)
qbc_commi_xgb_regr = XGBRegressor(**xgb_params)


# -------->  DATA INPUT  <------------------
df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/MLPlatform/data/auto_mpg.csv', header=0)
df_data = df_data.fillna(value=0)
arr_data = np.asarray(df_data)
trainingSetSizes = np.arange(seedSize, arr_data.shape[0], batchSize)
splits = trainingSetSizes / arr_data.shape[0]
sams = splits * arr_data.shape[0]
print(np.isnan(arr_data).any())


# -------->  MONTE CARLO SAMPLING (MCS) <------------------
np.random.seed(0)
pool = arr_data
np.random.shuffle(pool)
mcs_X_train_split = pool[:trainingSetSizes[0], :-1]
mcs_X_test_split = pool[trainingSetSizes[0]:, :-1]
mcs_y_train_split = pool[:trainingSetSizes[0], -1].reshape((-1, 1))
mcs_y_test_split = pool[trainingSetSizes[0]:, -1].reshape((-1, 1))


# -------->  ITERATIONS OVER DIFFERENT TEST SETS: TEST LOOP  <------------------
for test_loop in range(num_iters_test):
    for idx, batch in enumerate(trainingSetSizes):
        print(str(idx + 1) + '/' + str(len(trainingSetSizes)) + '  ' + str(test_loop + 1) + '/' + str(num_iters_test))

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(mcs_X_train_split)
        mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(mcs_y_train_split)

        # -------->  MODEL FITTING  <------------------
        mcs_mlp_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)
        mcs_xgb_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)
        mcs_rr_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)

        # -------->  PREDICTION <------------------
        print('trainingSetSize: ' + str(mcs_X_train_split.shape[0]))
        print('testSetSize: ' + str(mcs_X_test_split.shape[0]))
        print(mcs_X_test_split.shape, mcs_mean_X_train.shape, mcs_std_X_train.shape)

        mcs_mlp_y_norm = mcs_mlp_regr.predict((mcs_X_test_split - mcs_mean_X_train) / mcs_std_X_train)
        mcs_xgb_y_norm = mcs_xgb_regr.predict((mcs_X_test_split - mcs_mean_X_train) / mcs_std_X_train)
        mcs_rr_y_norm = mcs_rr_regr.predict((mcs_X_test_split - mcs_mean_X_train) / mcs_std_X_train)

        # -------->  DENORMALIZATION OF PREDICTIONS <------------------
        mcs_mlp_y = mcs_mlp_y_norm * mcs_std_y_train + mcs_mean_y_train
        mcs_xgb_y = mcs_xgb_y_norm * mcs_std_y_train + mcs_mean_y_train
        mcs_rr_y = mcs_rr_y_norm * mcs_std_y_train + mcs_mean_y_train

        # -------->  SAVING PREDICTED VALUES <------------------

        preds_rr_mcs = np.concatenate((preds_rr_mcs.reshape((-1, 1)), mcs_rr_y.reshape((-1, 1))))
        preds_mlp_mcs = np.concatenate((preds_mlp_mcs.reshape((-1, 1)), mcs_mlp_y.reshape((-1, 1))))
        preds_xgb_mcs = np.concatenate((preds_xgb_mcs.reshape((-1, 1)), mcs_xgb_y.reshape((-1, 1))))
        targets_rr_mcs = np.concatenate((targets_rr_mcs.reshape((-1, 1)), mcs_y_test_split.reshape((-1, 1))))
        targets_mlp_mcs = np.concatenate((targets_mlp_mcs.reshape((-1, 1)), mcs_y_test_split.reshape((-1, 1))))
        targets_xgb_mcs = np.concatenate((targets_xgb_mcs.reshape((-1, 1)), mcs_y_test_split.reshape((-1, 1))))


        if (idx + 1) == len(trainingSetSizes):
            np.random.seed(test_loop)
            pool = arr_data
            np.random.shuffle(pool)
            mcs_X_train_split = pool[:trainingSetSizes[0], :-1]
            mcs_X_test_split = pool[trainingSetSizes[0]:, :-1]
            mcs_y_train_split = pool[:trainingSetSizes[0], -1].reshape((-1, 1))
            mcs_y_test_split = pool[trainingSetSizes[0]:, -1].reshape((-1, 1))

        else:
            # -------->  PREPARING NEXT BATCH <------------------
            mcs_X_train_split = np.concatenate((mcs_X_train_split, mcs_X_test_split[:batchSize, :]))
            mcs_X_test_split = mcs_X_test_split[batchSize:, :]

            mcs_y_train_split = np.concatenate((mcs_y_train_split, mcs_y_test_split[:batchSize, :].reshape((-1,1))))
            mcs_y_test_split = mcs_y_test_split[batchSize:, :].reshape((-1,1))



# -------->  QUERY BY COMMITTEE (QBC) <------------------
np.random.seed(0)
pool = arr_data
np.random.shuffle(pool)
qbc_X_train_split = pool[:trainingSetSizes[0], :-1]
qbc_X_test_split = pool[trainingSetSizes[0]:, :-1]
qbc_y_train_split = pool[:trainingSetSizes[0], -1].reshape((-1, 1))
qbc_y_test_split = pool[trainingSetSizes[0]:, -1].reshape((-1, 1))

# -------->  ITERATIONS OVER DIFFERENT TEST SETS: TEST LOOP  <------------------
for test_loop in range(num_iters_test):
    for idx, batch in enumerate(trainingSetSizes):
        print(str(idx + 1) + '/' + str(len(trainingSetSizes)) + '  ' + str(test_loop + 1) + '/' + str(num_iters_test))

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)
        qbc_X_test_split_norm = (qbc_X_test_split - qbc_mean_X_train) / qbc_std_X_train
        qbc_y_test_split_norm = (qbc_y_test_split - qbc_mean_y_train) / qbc_std_y_train

        # -------->  MODEL FITTING  <------------------
        qbc_mlp_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)
        qbc_xgb_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)
        qbc_rr_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)

        # -------->  PREDICTION <------------------
        print('trainingSetSize: ' + str(qbc_X_train_split.shape[0]))
        print('testSetSize: ' + str(qbc_X_test_split.shape[0]))
        print(qbc_X_test_split.shape, qbc_mean_X_train.shape, qbc_std_X_train.shape)

        qbc_rr_y_norm = qbc_rr_regr.predict(qbc_X_test_split_norm)
        qbc_mlp_y_norm = qbc_mlp_regr.predict(qbc_X_test_split_norm)
        qbc_xgb_y_norm = qbc_xgb_regr.predict(qbc_X_test_split_norm)

        qbc_mlp_y = qbc_mlp_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_xgb_y = qbc_xgb_y_norm * qbc_std_y_train + qbc_mean_y_train
        qbc_rr_y = qbc_rr_y_norm * qbc_std_y_train + qbc_mean_y_train

        # -------->  SAVING PREDICTED VALUES <------------------
        preds_rr_qbc = np.concatenate((preds_rr_qbc.reshape((-1, 1)), qbc_rr_y.reshape((-1, 1))))
        preds_mlp_qbc = np.concatenate((preds_mlp_qbc.reshape((-1, 1)), qbc_mlp_y.reshape((-1, 1))))
        preds_xgb_qbc = np.concatenate((preds_xgb_qbc.reshape((-1, 1)), qbc_xgb_y.reshape((-1, 1))))
        targets_rr_qbc = np.concatenate((targets_rr_qbc.reshape((-1, 1)), qbc_y_test_split.reshape((-1, 1))))
        targets_mlp_qbc = np.concatenate((targets_mlp_qbc.reshape((-1, 1)), qbc_y_test_split.reshape((-1, 1))))
        targets_xgb_qbc = np.concatenate((targets_xgb_qbc.reshape((-1, 1)), qbc_y_test_split.reshape((-1, 1))))


        if (idx + 1) == len(trainingSetSizes):
            np.random.seed(test_loop)
            pool = arr_data
            np.random.shuffle(pool)
            qbc_X_train_split = pool[:trainingSetSizes[0], :-1]
            qbc_X_test_split = pool[trainingSetSizes[0]:, :-1]
            qbc_y_train_split = pool[:trainingSetSizes[0], -1].reshape((-1, 1))
            qbc_y_test_split = pool[trainingSetSizes[0]:, -1].reshape((-1, 1))


        else:
            print('AL creation of committee...')
            # --------> CREATION OF THE COMMITTEE) <------------------
            learner_list = []
            for i in range(numCommitteeMembers):
                learner_list.append(ActiveLearner(estimator=Ridge(**rr_params_v1),
                                                  X_training=qbc_X_train_split_norm,
                                                  y_training=qbc_y_train_split_norm,
                                                  bootstrap_init=True))

            committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

            # -------->  PREPARING NEXT BATCH <------------------
            for query in range(batchSize):
                # sample selection and committee training
                query_idx, query_instance = committee.query(qbc_X_test_split_norm)
                committee.teach(qbc_X_test_split_norm[query_idx], qbc_y_test_split_norm[query_idx], bootstrap=True)
                #committee.rebag()

                print('before: '+str(qbc_X_test_split.shape[0])+' '+str(qbc_X_train_split.shape[0]))
                qbc_X_train_split = np.append(qbc_X_train_split, qbc_X_test_split[query_idx], axis=0)
                qbc_y_train_split = np.append(qbc_y_train_split, qbc_y_test_split[query_idx], axis=0)
                qbc_X_train_split_norm = np.append(qbc_X_train_split_norm, qbc_X_test_split_norm[query_idx], axis=0)
                qbc_y_train_split_norm = np.append(qbc_y_train_split_norm, qbc_y_test_split_norm[query_idx], axis=0)

                qbc_X_test_split = np.delete(qbc_X_test_split, query_idx, axis=0)
                qbc_y_test_split = np.delete(qbc_y_test_split, query_idx, axis=0)
                qbc_X_test_split_norm = np.delete(qbc_X_test_split_norm, query_idx, axis=0)
                qbc_y_test_split_norm = np.delete(qbc_y_test_split_norm, query_idx, axis=0)

                print('after: ' + str(qbc_X_test_split.shape[0]) + ' ' + str(qbc_X_train_split.shape[0]))
                print(' ')




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


print("Done.")