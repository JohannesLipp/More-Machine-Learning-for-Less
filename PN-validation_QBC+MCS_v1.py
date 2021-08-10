import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

batchSize = 10     # does not work perfectly, as size_test_set cuts a certain perc from the data and batchsize decreases to 9
num_iters_test = 5  # choose desired number of test loop iterations
size_test_set = 0.1  # choose desired size of the test set
num_iters_train = 5  # choose desired number of training loop iterations


# -------->  CREATION OF LISTS CONTAINING THE PREDICTIONS  <------------------
preds_rr_mcs, preds_rr_qbc = [], []
preds_mlp_mcs, preds_mlp_qbc = [], []
preds_xgb_mcs, preds_xgb_qbc = [], []
targets_rr_mcs, targets_rr_qbc = [], []
targets_mlp_mcs, targets_mlp_qbc = [], []
targets_xgb_mcs, targets_xgb_qbc = [], []


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
    'alpha': 5, 'max_iter': 4, 'normalize': False, 'solver': 'lsqr', 'tol': 0.003
}
rr_params_v2 = {
    'alpha': 6, 'max_iter': 8, 'normalize': False, 'solver': 'lsqr', 'tol': 0.002
}
rr_params_v3 = {
    'alpha': 7, 'max_iter': 16, 'normalize': False, 'solver': 'lsqr', 'tol': 0.004
}
rr_params_v4 = {
    'alpha': 5, 'max_iter': 16, 'normalize': False, 'solver': 'lsqr', 'tol': 0.002
}
rr_params_v5 = {
    'alpha': 7, 'max_iter': 32, 'normalize': False, 'solver': 'lsqr', 'tol': 0.008
}
lhs_rr_regr = Ridge(**rr_params_v1)
mcs_rr_regr = Ridge(**rr_params_v1)
bbd_rr_regr = Ridge(**rr_params_v1)
ccd_rr_regr = Ridge(**rr_params_v1)
qbc_rr_regr = Ridge(**rr_params_v1)
qbc_commi_rr_regr = Ridge(**rr_params_v1)

# Multilayer Perceptron Neural Network MLP
mlp_params_v1 = {
    'solver': 'adam', 'hidden_layer_sizes': (60, 60), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 300
}
mlp_params_v2 = {
    'solver': 'adam', 'hidden_layer_sizes': (55, 55), 'activation': 'relu', 'tol': 1e-5
}
mlp_params_v3 = {
    'solver': 'adam', 'hidden_layer_sizes': (65, 65), 'activation': 'relu', 'tol': 1e-4
}

lhs_mlp_regr = MLPRegressor(**mlp_params_v1)
mcs_mlp_regr = MLPRegressor(**mlp_params_v1)
bbd_mlp_regr = MLPRegressor(**mlp_params_v1)
ccd_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_commi_mlp_regr = MLPRegressor(**mlp_params_v1)

# eXtreme Gradient Boosting XGB
xgb_params = {
    'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
    'booster': 'gbtree', 'n_jobs': -1
}

lhs_xgb_regr = XGBRegressor(**xgb_params)
mcs_xgb_regr = XGBRegressor(**xgb_params)
bbd_xgb_regr = XGBRegressor(**xgb_params)
ccd_xgb_regr = XGBRegressor(**xgb_params)
qbc_xgb_regr = XGBRegressor(**xgb_params)
qbc_commi_xgb_regr = XGBRegressor(**xgb_params)

# initializing Committee members
num_committee_members = 5

# -------->  DATA INPUT  <------------------
df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/MLPlatform/data/auto_mpg.csv', header=0)
df_data = df_data.fillna(value=0)
arr_data = np.asarray(df_data)
frac = batchSize / arr_data.shape[0]
splits = np.arange(frac, 1, frac)
sams = splits * arr_data.shape[0]


# -------->  ITERATIONS OVER DIFFERENT TEST SETS: TEST LOOP  <------------------

for test_loop in range(num_iters_test):
    X_train, X_test, y_train, y_test = train_test_split(arr_data[:, :-1], arr_data[:, -1], test_size=size_test_set)
    print(len(y_test))
    # -------->  ITERATIONS OVER TRAINING SET SIZES: LEARNING CURVE LOOP  <------------------
    training_set_sizes = [int(np.round(len(y_train) * i)) for i in splits]
    #print(training_set_sizes)
    print(y_train.shape)

    # -------->  ITERATIONS OVER TRAINING SETS FOR HOMOGENIC TRAINING RESULTS: TRAIN LOOP <------------------
    for train_loop in range(num_iters_train):

        for lc_loop in range(splits.shape[0]):
            print(str(test_loop) + '.' + str(train_loop) + '.' + str(lc_loop))


            # -------->  MONTE CARLO SAMPLING (MCS) <------------------
            mcs_X_train_split, mcs_X_test_split, mcs_y_train_split, mcs_y_test_split = \
                train_test_split(X_train, y_train, train_size=int(training_set_sizes[lc_loop]))


            # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
            mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(mcs_X_train_split)
            mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(mcs_y_train_split)
            print(mcs_y_train_split_norm.shape)

            # -------->  MODEL FITTING  <------------------
            mcs_mlp_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)
            mcs_xgb_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)
            mcs_rr_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)

            # -------->  PREDICTION <------------------
            mcs_mlp_y_norm = mcs_mlp_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
            mcs_xgb_y_norm = mcs_xgb_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
            mcs_rr_y_norm = mcs_rr_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)

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

            # -------->  END OF TRAINING LOOP <------------------
        # -------->  END OF LEARNING CURVE LOOP <------------------


    # -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
    print('calculating QBC...')
    '''
    # -------->  ITERATIONS OVER TRAINING SETS FOR HOMOGENIC TRAINING RESULTS: TRAIN LOOP <------------------
    for train_loop in range(num_iters_train):
        print('AL initiation first training set size')

        # initial training data for committee and normalization
        n_initial = training_set_sizes[0]
        learner_list = []
        X_pool = X_train.copy()
        y_pool = y_train.copy()
        indices = np.random.randint(low=X_pool.shape[0], size=n_initial)
        X_train_init = X_pool[indices]
        y_train_init = y_pool[indices]

        X_pool = np.delete(X_pool, indices, axis=0)
        y_pool = np.delete(y_pool, indices, axis=0)

        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(X_train_init)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(y_train_init)
        print('AL creation of committee...')
        # creation of committee
        learner_list.append(ActiveLearner(
            estimator=Ridge(**rr_params_v1), X_training=qbc_X_train_split_norm,
            y_training=qbc_y_train_split_norm.ravel()))
        learner_list.append(ActiveLearner(
            estimator=Ridge(**rr_params_v2), X_training=qbc_X_train_split_norm,
            y_training=qbc_y_train_split_norm.ravel()))
        learner_list.append(ActiveLearner(
            estimator=Ridge(**rr_params_v3), X_training=qbc_X_train_split_norm,
            y_training=qbc_y_train_split_norm.ravel()))
        learner_list.append(ActiveLearner(
            estimator=Ridge(**rr_params_v4), X_training=qbc_X_train_split_norm,
            y_training=qbc_y_train_split_norm.ravel()))
        learner_list.append(ActiveLearner(
            estimator=Ridge(**rr_params_v5), X_training=qbc_X_train_split_norm,
            y_training=qbc_y_train_split_norm.ravel()))
        committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

        print('AL pre-training of models...')
        # pre-training of models
        qbc_rr_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)
        qbc_mlp_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
        qbc_xgb_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)

        # predicition
        qbc_rr_y_std = qbc_rr_regr.predict((X_test - qbc_mean_X_train) / qbc_std_X_train)
        qbc_mlp_y_std = qbc_mlp_regr.predict((X_test - qbc_mean_X_train) / qbc_std_X_train)
        qbc_xgb_y_std = qbc_xgb_regr.predict((X_test - qbc_mean_X_train) / qbc_std_X_train)

        qbc_rr_y = qbc_rr_y_std * qbc_std_y_train + qbc_mean_y_train
        qbc_mlp_y = qbc_mlp_y_std * qbc_std_y_train + qbc_mean_y_train
        qbc_xgb_y = qbc_xgb_y_std * qbc_std_y_train + qbc_mean_y_train

        preds_rr_qbc.append(qbc_rr_y)
        preds_mlp_qbc.append(qbc_mlp_y)
        preds_xgb_qbc.append(qbc_xgb_y)

        qbc_X_train_split = qbc_X_train_split_norm * qbc_std_X_train + qbc_mean_X_train
        qbc_y_train_split = qbc_y_train_split_norm * qbc_std_y_train + qbc_mean_y_train

        print('AL query loop...')
        qbc_learning_intervals = np.asarray(training_set_sizes) - training_set_sizes[0]
        n_queries = qbc_learning_intervals[-1]
        for idx in range(n_queries):
            # sample selection and committee training
            query_idx, query_instance = committee.query(X_pool)
            committee.teach(X_pool[query_idx], y_pool[query_idx], bootstrap=True)

            # addition of sample to the training set
            qbc_X_train_split = np.concatenate((qbc_X_train_split, X_pool[query_idx]), axis=0)
            qbc_y_train_split = np.concatenate((qbc_y_train_split, y_pool[query_idx]), axis=0)
            print(len(qbc_X_train_split))

            if idx in qbc_learning_intervals:
                print('AL test: ' + str(test_loop) + ' lc: ' + str(idx + training_set_sizes[0])
                      + ' train: ' + str(train_loop))
                # normalization of local training set
                qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
                qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)

                # model fitting
                qbc_rr_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)
                qbc_mlp_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())
                qbc_xgb_regr.fit(qbc_X_train_split_norm, qbc_y_train_split_norm)

                # predicition
                qbc_rr_y_std = qbc_rr_regr.predict((X_test - qbc_mean_X_train) / qbc_std_X_train)
                qbc_mlp_y_std = qbc_mlp_regr.predict((X_test - qbc_mean_X_train) / qbc_std_X_train)
                qbc_xgb_y_std = qbc_xgb_regr.predict((X_test - qbc_mean_X_train) / qbc_std_X_train)

                qbc_rr_y = qbc_rr_y_std * qbc_std_y_train + qbc_mean_y_train
                qbc_mlp_y = qbc_mlp_y_std * qbc_std_y_train + qbc_mean_y_train
                qbc_xgb_y = qbc_xgb_y_std * qbc_std_y_train + qbc_mean_y_train

                preds_rr_qbc.append(qbc_rr_y)
                preds_mlp_qbc.append(qbc_mlp_y)
                preds_xgb_qbc.append(qbc_xgb_y)

                # denormalization
                qbc_X_train_split = qbc_X_train_split_norm * qbc_std_X_train + qbc_mean_X_train
                qbc_y_train_split = qbc_y_train_split_norm * qbc_std_y_train + qbc_mean_y_train

            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
    '''
# -------->  END OF SAMPLING AND PREDICTION CYCLE <------------------


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

print("Done.")
