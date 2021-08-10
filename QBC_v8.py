import numpy as np
import pandas as pd
from pyDOE import lhs
from pyDOE2 import bbdesign
from pyDOE2 import ccdesign
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------
doc = 1  # choose desired DOC
factor_ranges = [11, 11, 11, 5, 11]

num_dims = [2, 4, 5, 5]
num_iters_test = 3  # choose desired number of test loop iterations
size_test_set = 0.1  # choose desired size of the test set
num_iters_train = 3  # choose desired number of training loop iterations

if doc == 1:
    splits = np.asarray([0.04, 0.05, 0.06, 0.068, 0.07, 0.08, 0.09, 0.11, 0.14, 0.186, 0.28, 0.46, 0.65, 0.83])
elif doc == 2:
    splits = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.0501, 0.1002, 0.2003, 0.501])
elif doc == 3 or doc == 4:
    splits = np.asarray([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.00501,
                         0.01002, 0.02003, 0.05009, 0.10018, 0.20035, 0.50088])

# resulting training set sizes doc1 = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 70, 90, 107]


# -------->  CREATION OF LISTS CONTAINING THE PREDICTIONS  <------------------
preds_rr_mcs, preds_rr_lhs, preds_rr_bbd, preds_rr_ccd, preds_rr_qbc = [], [], [], [], []
preds_mlp_mcs, preds_mlp_lhs, preds_mlp_bbd, preds_mlp_ccd, preds_mlp_qbc = [], [], [], [], []
preds_xgb_mcs, preds_xgb_lhs, preds_xgb_bbd, preds_xgb_ccd, preds_xgb_qbc = [], [], [], [], []
targets_rr_mcs, targets_rr_lhs, targets_rr_bbd, targets_rr_ccd, targets_rr_qbc = [], [], [], [], []
targets_mlp_mcs, targets_mlp_lhs, targets_mlp_bbd, targets_mlp_ccd, targets_mlp_qbc = [], [], [], [], []
targets_xgb_mcs, targets_xgb_lhs, targets_xgb_bbd, targets_xgb_ccd, targets_xgb_qbc = [], [], [], [], []


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

def latin_hypercube_sampler(X, y, num_dimensions, num_samples):

    hypercube = lhs(num_dimensions, num_samples, 'center')

    hypercube *= factor_ranges[:num_dimensions]
    hypercube = np.floor(hypercube)

    df_X_entire = pd.DataFrame(X)
    df_hypercube = pd.DataFrame(hypercube)
    df_concat = pd.concat((df_X_entire, df_hypercube))
    features = X[df_concat.duplicated(keep=False).iloc[:-len(df_hypercube)]]
    targets = y[df_concat.duplicated(keep=False).iloc[:-len(df_hypercube)]]
    indices = df_concat.duplicated(keep=False).iloc[:-len(df_hypercube)]

    if len(features) != num_samples:
        df_features = pd.DataFrame(features)
        df_targets = pd.DataFrame(targets)
        df_split = pd.concat((df_features, df_targets), axis=1)

        df_y_entire = pd.DataFrame(y)
        df_entire = pd.concat((df_X_entire, df_y_entire), axis=1)

        df_concat2 = pd.concat((df_entire, df_split))
        df_missing_samples = df_concat2.drop_duplicates(keep=False)  # .iloc[:-len(df_X_train_split)]
        df_samples_rest = df_missing_samples.sample(n=(num_samples - len(features)))

        features = np.asarray(pd.concat((df_features, df_samples_rest.iloc[:, :-1])))
        targets = np.asarray(pd.concat((df_targets, df_samples_rest.iloc[:, -1])))
        targets = np.ravel(targets)

    return features, targets, indices

# -------->  MACHINE LEARNING MODELS (HYPERPARAMETERS, COMMITTEE FOR QBC) <------------------

# Ridge Regression RR
rr_params = {
    'alpha': 5, 'max_iter': 4, 'normalize': False, 'solver': 'lsqr', 'tol': 0.003
    }
lhs_rr_regr = Ridge(**rr_params)
mcs_rr_regr = Ridge(**rr_params)
bbd_rr_regr = Ridge(**rr_params)
ccd_rr_regr = Ridge(**rr_params)
qbc_rr_regr = Ridge(**rr_params)
qbc_commi_rr_regr = Ridge(**rr_params)

# Multilayer Perceptron Neural Network MLP
mlp_params = {
    'solver': 'adam', 'hidden_layer_sizes': (60, 60), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 300
    }

lhs_mlp_regr = MLPRegressor(**mlp_params)
mcs_mlp_regr = MLPRegressor(**mlp_params)
bbd_mlp_regr = MLPRegressor(**mlp_params)
ccd_mlp_regr = MLPRegressor(**mlp_params)
qbc_mlp_regr = MLPRegressor(**mlp_params)
qbc_commi_mlp_regr = MLPRegressor(**mlp_params)

# eXtreme Gradient Boosting XGB
xgb_params = {
    'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
    'booster': 'gbtree'
    }

lhs_xgb_regr = XGBRegressor(**xgb_params)
mcs_xgb_regr = XGBRegressor(**xgb_params)
bbd_xgb_regr = XGBRegressor(**xgb_params)
ccd_xgb_regr = XGBRegressor(**xgb_params)
qbc_xgb_regr = XGBRegressor(**xgb_params)
qbc_commi_xgb_regr = XGBRegressor(**xgb_params)

# initializing Committee members
num_committee_members = 3


# -------->  DATA INPUT  <------------------

if doc == 1:
    df_data = pd.read_csv(
        '/Users/philippnoodt/Studium/RWTH/B.Sc. Maschinenbau Energietechnik/BA/Python_BA/ba-philipp-code/Simulated Data/'
        'data_preprocessed_DOC1.csv', usecols=[0, 1, 2], header=0)
    # df_data column names=['x1', 'x2', 'Energy consumption']


elif doc == 2:
    df_data = pd.read_csv(
        '/Users/philippnoodt/Studium/RWTH/B.Sc. Maschinenbau Energietechnik/BA/Python_BA/ba-philipp-code/Simulated Data/'
        'data_preprocessed_DOC2.csv', usecols=[0, 1, 2, 3, 4], header=0)
    # df_data column names=['x1', 'x2', 'x3', 'x4', 'Energy consumption']

elif doc == 3:
    df_data = pd.read_csv(
        '/Users/philippnoodt/Studium/RWTH/B.Sc. Maschinenbau Energietechnik/BA/Python_BA/ba-philipp-code/Simulated Data/'
        'data_preprocessed_DOC3_v2.csv', usecols=[0, 1, 2, 3, 4, 5], header=0)
    # df_data column names=['x1', 'x2', 'x3', 'x4', 'x5', 'Energy consumption']

elif doc == 4:
    df_data = pd.read_csv(
        '/Users/philippnoodt/Studium/RWTH/B.Sc. Maschinenbau Energietechnik/BA/Python_BA/ba-philipp-code/Simulated Data/'
        'data_preprocessed_DOC4_v2.csv', usecols=[0, 1, 2, 3, 4, 5], header=0)
    # df_data column names=['x1', 'x2', 'x3', 'x4', 'x5', 'Energy consumption']

else:
    print('DOC invalid')

arr_data = np.asarray(df_data)


# -------->  ITERATIONS OVER DIFFERENT TEST SETS: TEST LOOP  <------------------

for test_loop in range(num_iters_test):
    X_train, X_test, y_train, y_test = train_test_split(arr_data[:, :-1], arr_data[:, -1], test_size=size_test_set)
    print(len(y_test))
    # -------->  ITERATIONS OVER TRAINING SET SIZES: LEARNING CURVE LOOP  <------------------
    training_set_sizes = [int(np.round(len(y_train) * i)) for i in splits]
    # training_set_sizes = sams
    print(training_set_sizes)

    for lc_loop in range(splits.shape[0]):
        num_dim = X_train.shape[1]

        # -------->  ITERATIONS OVER TRAINING SETS FOR HOMOGENIC TRAINING RESULTS: TRAIN LOOP <------------------
        for train_loop in range(num_iters_train):
            print(str(test_loop) + '.' + str(lc_loop) + '.' + str(train_loop))

            # -------->  LATIN HYPERCUBE SAMPLING (LHS) <------------------
            lhs_X_train_split, lhs_y_train_split, indices = latin_hypercube_sampler(
                X=X_train, y=y_train, num_dimensions=num_dim, num_samples=training_set_sizes[lc_loop])

            # -------->  MONTE CARLO SAMPLING (MCS) <------------------
            mcs_X_train_split, mcs_X_test_split, mcs_y_train_split, mcs_y_test_split = \
                train_test_split(X_train, y_train, train_size=int(training_set_sizes[lc_loop]))


            # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
            lhs_mean_X_train, lhs_std_X_train, lhs_X_train_split_norm = normalize(lhs_X_train_split)
            lhs_mean_y_train, lhs_std_y_train, lhs_y_train_split_norm = normalize(lhs_y_train_split)

            mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(mcs_X_train_split)
            mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(mcs_y_train_split)


            # -------->  MODEL FITTING  <------------------
            lhs_mlp_regr.fit(lhs_X_train_split_norm, lhs_y_train_split_norm.ravel())
            lhs_xgb_regr.fit(lhs_X_train_split_norm, lhs_y_train_split_norm)
            lhs_rr_regr.fit(lhs_X_train_split_norm, lhs_y_train_split_norm)

            mcs_mlp_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
            mcs_xgb_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)
            mcs_rr_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm)

            # -------->  PREDICTION <------------------
            lhs_mlp_y_norm = lhs_mlp_regr.predict((X_test - lhs_mean_X_train) / lhs_std_X_train)
            lhs_xgb_y_norm = lhs_xgb_regr.predict((X_test - lhs_mean_X_train) / lhs_std_X_train)
            lhs_rr_y_norm = lhs_rr_regr.predict((X_test - lhs_mean_X_train) / lhs_std_X_train)

            lhs_mlp_y = lhs_mlp_y_norm * lhs_std_y_train + lhs_mean_y_train
            lhs_xgb_y = lhs_xgb_y_norm * lhs_std_y_train + lhs_mean_y_train
            lhs_rr_y = lhs_rr_y_norm * lhs_std_y_train + lhs_mean_y_train

            mcs_mlp_y_norm = mcs_mlp_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
            mcs_xgb_y_norm = mcs_xgb_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
            mcs_rr_y_norm = mcs_rr_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)

            mcs_mlp_y = mcs_mlp_y_norm * mcs_std_y_train + mcs_mean_y_train
            mcs_xgb_y = mcs_xgb_y_norm * mcs_std_y_train + mcs_mean_y_train
            mcs_rr_y = mcs_rr_y_norm * mcs_std_y_train + mcs_mean_y_train

            # -------->  ACTIVE SAMPLING AND PREDICTION  <------------------


            # -------->  SAVING PREDICTED VALUES <------------------
            preds_rr_lhs.append(lhs_rr_y)
            preds_mlp_lhs.append(lhs_mlp_y)
            preds_xgb_lhs.append(lhs_xgb_y)
            targets_rr_lhs.append(y_test)
            targets_mlp_lhs.append(y_test)
            targets_xgb_lhs.append(y_test)

            preds_rr_mcs.append(mcs_rr_y)
            preds_mlp_mcs.append(mcs_mlp_y)
            preds_xgb_mcs.append(mcs_xgb_y)
            targets_rr_mcs.append(y_test)
            targets_mlp_mcs.append(y_test)
            targets_xgb_mcs.append(y_test)

            # -------->  END OF TRAINING LOOP <------------------
        # -------->  END OF LEARNING CURVE LOOP <------------------

    # -------->  BBD AND CCD DOEs  <------------------
    print('calculating BBD and CCD...')
    df_X_train_entire = pd.DataFrame(X_train)  # is already created before, I just do it again though

    # -------->  BBD  <------------------
    if doc != 1:
        df_bbd = pd.DataFrame(bbdesign(num_dims[doc - 1]))
        df_bbd.drop_duplicates(inplace=True)
        bbd = np.asarray(df_bbd)
        bbd -= np.min(bbd)
        bbd /= np.max(bbd)
        bbd *= factor_ranges[:num_dims[doc - 1]]
        bbd = np.floor(bbd)
        bbd[bbd == 11] = 10

        df_bbd = pd.DataFrame(bbd)
        df_bbd_concat = pd.concat((df_X_train_entire, df_bbd))
        bbd_X_train_split, bbd_y_train_split = X_train[df_bbd_concat.duplicated(keep=False).iloc[:-len(df_bbd)]], \
                                               y_train[df_bbd_concat.duplicated(keep=False).iloc[:-len(df_bbd)]]

        if len(bbd_X_train_split) != bbd.shape[0]:
            df_X_train_split = pd.DataFrame(bbd_X_train_split)
            df_y_train_split = pd.DataFrame(bbd_y_train_split)
            df_train_split = pd.concat((df_X_train_split, df_y_train_split), axis=1)

            df_y_train_entire = pd.DataFrame(y_train)
            df_train_entire = pd.concat((df_X_train_entire, df_y_train_entire), axis=1)

            df_concat2 = pd.concat((df_train_entire, df_train_split))
            df_missing_samples = df_concat2.drop_duplicates(keep=False)  # .iloc[:-len(df_X_train_split)]
            df_samples_rest = df_missing_samples.sample(n=(bbd.shape[0] - len(bbd_X_train_split)))

            bbd_X_train_split = np.asarray(pd.concat((df_X_train_split, df_samples_rest.iloc[:, :-1])))
            bbd_y_train_split = np.asarray(pd.concat((df_y_train_split, df_samples_rest.iloc[:, -1])))
            bbd_y_train_split = np.ravel(bbd_y_train_split)

    # -------->  CCD  <------------------
    df_ccd = pd.DataFrame(ccdesign(num_dims[doc - 1]))
    df_ccd.drop_duplicates(inplace=True)
    ccd = np.asarray(df_ccd)
    ccd -= np.min(ccd)
    ccd /= np.max(ccd)
    ccd *= factor_ranges[:num_dims[doc - 1]]
    ccd = np.floor(ccd)
    ccd[ccd == 11] = 10

    df_ccd = pd.DataFrame(ccd)
    df_ccd_concat = pd.concat((df_X_train_entire, df_ccd))
    ccd_X_train_split, ccd_y_train_split = X_train[df_ccd_concat.duplicated(keep=False).iloc[:-len(df_ccd)]], \
                                           y_train[df_ccd_concat.duplicated(keep=False).iloc[:-len(df_ccd)]]

    if len(ccd_X_train_split) != ccd.shape[0]:
        df_X_train_split = pd.DataFrame(ccd_X_train_split)
        df_y_train_split = pd.DataFrame(ccd_y_train_split)
        df_train_split = pd.concat((df_X_train_split, df_y_train_split), axis=1)

        df_y_train_entire = pd.DataFrame(y_train)
        df_train_entire = pd.concat((df_X_train_entire, df_y_train_entire), axis=1)

        df_concat2 = pd.concat((df_train_entire, df_train_split))
        df_missing_samples = df_concat2.drop_duplicates(keep=False)  # .iloc[:-len(df_X_train_split)]
        df_samples_rest = df_missing_samples.sample(n=(ccd.shape[0] - len(ccd_X_train_split)))

        ccd_X_train_split = np.asarray(pd.concat((df_X_train_split, df_samples_rest.iloc[:, :-1])))
        ccd_y_train_split = np.asarray(pd.concat((df_y_train_split, df_samples_rest.iloc[:, -1])))
        ccd_y_train_split = np.ravel(ccd_y_train_split)

    # --------> NORMALIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <----------------
    # if DOC1: mark every operation with bbd_xx, the bbd code must not run.
    # Then follow futher instructions in the comments
    if doc != 1:
        bbd_mean_X_train, bbd_std_X_train, bbd_X_train_split_norm = normalize(bbd_X_train_split)
        bbd_mean_y_train, bbd_std_y_train, bbd_y_train_split_norm = normalize(bbd_y_train_split)

    ccd_mean_X_train, ccd_std_X_train, ccd_X_train_split_norm = normalize(ccd_X_train_split)
    ccd_mean_y_train, ccd_std_y_train, ccd_y_train_split_norm = normalize(ccd_y_train_split)

    # -------->  MODEL FITTING  <------------------
    if doc != 1:
        bbd_mlp_regr.fit(bbd_X_train_split_norm, bbd_y_train_split_norm.ravel())
        bbd_xgb_regr.fit(bbd_X_train_split_norm, bbd_y_train_split_norm)
        bbd_rr_regr.fit(bbd_X_train_split_norm, bbd_y_train_split_norm)

    ccd_mlp_regr.fit(ccd_X_train_split_norm, ccd_y_train_split_norm.ravel())
    ccd_xgb_regr.fit(ccd_X_train_split_norm, ccd_y_train_split_norm)
    ccd_rr_regr.fit(ccd_X_train_split_norm, ccd_y_train_split_norm)

    # -------->  PREDICTION <------------------
    if doc != 1:
        bbd_mlp_y_std = bbd_mlp_regr.predict((X_test - bbd_mean_X_train) / bbd_std_X_train)
        bbd_xgb_y_std = bbd_xgb_regr.predict((X_test - bbd_mean_X_train) / bbd_std_X_train)
        bbd_rr_y_std = bbd_rr_regr.predict((X_test - bbd_mean_X_train) / bbd_std_X_train)

        bbd_mlp_y = bbd_mlp_y_std * bbd_std_y_train + bbd_mean_y_train
        bbd_xgb_y = bbd_xgb_y_std * bbd_std_y_train + bbd_mean_y_train
        bbd_rr_y = bbd_rr_y_std * bbd_std_y_train + bbd_mean_y_train

    ccd_mlp_y_std = ccd_mlp_regr.predict((X_test - ccd_mean_X_train) / ccd_std_X_train)
    ccd_xgb_y_std = ccd_xgb_regr.predict((X_test - ccd_mean_X_train) / ccd_std_X_train)
    ccd_rr_y_std = ccd_rr_regr.predict((X_test - ccd_mean_X_train) / ccd_std_X_train)

    ccd_mlp_y = ccd_mlp_y_std * ccd_std_y_train + ccd_mean_y_train
    ccd_xgb_y = ccd_xgb_y_std * ccd_std_y_train + ccd_mean_y_train
    ccd_rr_y = ccd_rr_y_std * ccd_std_y_train + ccd_mean_y_train

    # -------->  SAVING PREDICTED VALUES <------------------
    if doc != 1:
        preds_rr_bbd.append(bbd_rr_y)
        preds_mlp_bbd.append(bbd_mlp_y)
        preds_xgb_bbd.append(bbd_xgb_y)
        targets_rr_bbd.append(y_test)
        targets_mlp_bbd.append(y_test)
        targets_xgb_bbd.append(y_test)

    preds_rr_ccd.append(ccd_rr_y)
    preds_mlp_ccd.append(ccd_mlp_y)
    preds_xgb_ccd.append(ccd_xgb_y)
    targets_rr_ccd.append(y_test)
    targets_mlp_ccd.append(y_test)
    targets_xgb_ccd.append(y_test)
    

    # -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
    print('calculating QBC...')

    # -------->  ITERATIONS OVER TRAINING SETS FOR HOMOGENIC TRAINING RESULTS: TRAIN LOOP <------------------
    for train_loop in range(num_iters_train):
        print('AL initiation first training set size')

        # initial training data for committee and normalization
        n_initial = training_set_sizes[0]
        learner_list = []
        X_pool = X_train.copy()
        y_pool = y_train.copy()
        X_train_init, y_train_init, indices = latin_hypercube_sampler(
            X=X_pool, y=y_pool, num_dimensions=X_pool.shape[1], num_samples=n_initial)
        X_pool = X_pool[~indices]
        y_pool = y_pool[~indices]

        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(X_train_init)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(y_train_init)
        print('AL creation of committee...')
        # creation of committee
        learner_list.append(ActiveLearner(
            estimator=Ridge(**rr_params), X_training=qbc_X_train_split_norm, y_training=qbc_y_train_split_norm.ravel()))
        learner_list.append(ActiveLearner(
            estimator=MLPRegressor(**mlp_params), X_training=qbc_X_train_split_norm, y_training=qbc_y_train_split_norm.ravel()))
        learner_list.append(ActiveLearner(
            estimator=XGBRegressor(**xgb_params), X_training=qbc_X_train_split_norm, y_training=qbc_y_train_split_norm.ravel()))
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
            committee.teach(X_pool[query_idx], y_pool[query_idx])

            # addition of sample to the training set and re-normalization
            qbc_X_train_split = np.concatenate((qbc_X_train_split,X_pool[query_idx]), axis=0)
            qbc_y_train_split = np.concatenate((qbc_y_train_split, y_pool[query_idx]), axis=0)

            qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
            qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)

            if idx in qbc_learning_intervals:
                print('AL test: ' + str(test_loop) + ' lc: ' + str(idx+ training_set_sizes[0])
                      + ' train: ' + str(train_loop))
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

                qbc_X_train_split = qbc_X_train_split_norm * qbc_std_X_train + qbc_mean_X_train
                qbc_y_train_split = qbc_y_train_split_norm * qbc_std_y_train + qbc_mean_y_train


            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
# -------->  END OF SAMPLING AND PREDICTION CYCLE <------------------


# -------->  SAVING THE RESULTS  <------------------
print("Saving...")

np.savetxt("preds_rr_mcs.csv", np.ravel(preds_rr_mcs), delimiter=",")
np.savetxt("preds_mlp_mcs.csv", np.ravel(preds_mlp_mcs), delimiter=",")
np.savetxt("preds_xgb_mcs.csv", np.ravel(preds_xgb_mcs), delimiter=",")

np.savetxt("preds_rr_lhs.csv", np.ravel(preds_rr_lhs), delimiter=",")
np.savetxt("preds_mlp_lhs.csv", np.ravel(preds_mlp_lhs), delimiter=",")
np.savetxt("preds_xgb_lhs.csv", np.ravel(preds_xgb_lhs), delimiter=",")

# np.savetxt("preds_rr_bbd.csv", np.ravel(preds_rr_bbd), delimiter=",")
# np.savetxt("preds_mlp_bbd.csv", np.ravel(preds_mlp_bbd), delimiter=",")
# np.savetxt("preds_xgb_bbd.csv", np.ravel(preds_xgb_bbd), delimiter=",")

np.savetxt("preds_rr_ccd.csv", np.ravel(preds_rr_ccd), delimiter=",")
np.savetxt("preds_mlp_ccd.csv", np.ravel(preds_mlp_ccd), delimiter=",")
np.savetxt("preds_xgb_ccd.csv", np.ravel(preds_xgb_ccd), delimiter=",")

np.savetxt("targets_rr_mcs.csv", np.ravel(targets_rr_mcs), delimiter=",")
np.savetxt("targets_mlp_mcs.csv", np.ravel(targets_mlp_mcs), delimiter=",")
np.savetxt("targets_xgb_mcs.csv", np.ravel(targets_xgb_mcs), delimiter=",")

np.savetxt("targets_rr_lhs.csv", np.ravel(targets_rr_lhs), delimiter=",")
np.savetxt("targets_mlp_lhs.csv", np.ravel(targets_mlp_lhs), delimiter=",")
np.savetxt("targets_xgb_lhs.csv", np.ravel(targets_xgb_lhs), delimiter=",")

# np.savetxt("targets_rr_bbd.csv", np.ravel(targets_rr_bbd), delimiter=",")
# np.savetxt("targets_mlp_bbd.csv", np.ravel(targets_mlp_bbd), delimiter=",")
# np.savetxt("targets_xgb_bbd.csv", np.ravel(targets_xgb_bbd), delimiter=",")

np.savetxt("targets_rr_ccd.csv", np.ravel(targets_rr_ccd), delimiter=",")
np.savetxt("targets_mlp_ccd.csv", np.ravel(targets_mlp_ccd), delimiter=",")
np.savetxt("targets_xgb_ccd.csv", np.ravel(targets_xgb_ccd), delimiter=",")

np.savetxt("preds_rr_qbc.csv", np.ravel(preds_rr_qbc), delimiter=",")
np.savetxt("preds_mlp_qbc.csv", np.ravel(preds_mlp_qbc), delimiter=",")
np.savetxt("preds_xgb_qbc.csv", np.ravel(preds_xgb_qbc), delimiter=",")

'''
out_file = "2018-11-13_MLP-XGB-RR_LHS-MCS-CCD_DOC1.csv"
print("Saving to: " + out_file)
df_saved_results.to_csv(out_file)
'''
print("Done.")
