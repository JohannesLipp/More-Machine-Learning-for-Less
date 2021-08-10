import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark import sql
# from joblib import load
from pyDOE import lhs
import pickle
import lightgbm
from sklearn.metrics import r2_score
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor

print()

startTime1 = datetime.now()
# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

# batchSize = 1  # size of the training set increments
# seedSize = 20  # size of the initial training set
# num_iters_test = 20  # 10  # choose desired number of cv iterations
# numCommitteeMembers = 3  # 5  # choose number of committee members for the QBC
#
# # -------->  PARAMETERS DEFINING THE TRAINING AND VALIDATION SPACE  <------------------
# size_test_set = 1000  # 0 #size of the dataset to be generated using the reference ML Model for testing
# size_train_set = 200  # 00 #size of the dataset to be generated using the reference ML Model for training


# batchSize = 10
# seedSize = 300
# num_iters_test = 5
# numCommitteeMembers = 3
# size_test_set = 1000
# size_train_set = 500

batchSize = 10
seedSize = 300
num_iters_test = 3
numCommitteeMembers = 3
size_test_set = 1000
size_train_set = 100

committee_predictor = True
# # Combined_Cycle_Power_Plant
# actual_lows = {'AT': 2, 'V': 30, 'AP': 993, 'RH': 30}  # , 'PE':[425]}
# actual_highs = {'AT': 35, 'V': 80, 'AP': 1033, 'RH': 100}  # , 'PE':[495]}
# variables = ['AT', 'V', 'AP', 'RH', 'PE']
# model_name = '/PowerPlant_ML.txt'

actual_lows = {'theta1': -1.8, 'theta2': -1.8, 'theta3': -1.8, 'thetad1': -1.8, 'thetad2': -1.8, 'thetad3': -1.8,
              'tau1':-0.5, 'tau2':-0.5}
actual_highs = {'theta1': 1.8, 'theta2': 1.8, 'theta3': 1.8, 'thetad1': 1.8, 'thetad2': 1.8, 'thetad3': 1.8,
               'tau1':0.5, 'tau2':0.5}
variables = ['theta1', 'theta2', 'theta3', 'thetad1', 'thetad2', 'thetad3', 'tau1', 'tau2', 'thetadd3']
model_name = '/pumadyn_8nm_ML.txt'


# actual_lows = {'theta1': -2.3555153, 'theta2': -2.3559486, 'theta3': -2.3556861000000002, 'theta4': -2.3554262,
#                 'theta5': -2.355844, 'theta6': -2.3558267999999996, 'thetad1': -2.3556139, 'thetad2': -2.3550772999999996,
#                 'thetad3': -2.3556497000000003,'thetad4': -2.3553389,'thetad5': -2.3550476000000002,'thetad6': -2.3561332999999998,
#                 'tau1': -74.990634, 'tau2': -74.937538, 'tau3': -74.978278, 'tau4': -74.999533, 'tau5': -74.984873,
#                 'dm1': 0.25026488,'dm2': 0.25041709, 'dm3': 0.25022193, 'dm4': 0.25007437, 'dm5': 0.25014939, 'da1': 0.25043561,
#                 'da2': 0.25057928, 'da3': 0.25008744, 'da4': 0.2501664, 'da5': 0.25061589, 'db1': 0.25005398,
#                 'db2': 0.25001159, 'db3': 0.25024083, 'db4': 0.25010881, 'db5': 0.2514393}
# actual_highs = {'theta1': 2.35, 'theta2': 2.355, 'theta3': 2.355, 'theta4': 2.355,
#                 'theta5': 2.35,'theta6': 2.35,'thetad1': 2.35,'thetad2': 2.3540761000000003,'thetad3': 2.3547369,
#                 'thetad4': 2.3557772,'thetad5': 2.3557997999999998,'thetad6': 2.3558617,'tau1': 74.985591,'tau2': 74.967958,
#                 'tau3': 74.986797,'tau4': 74.99699100000001,'tau5': 74.995852,'dm1': 2.4999799,'dm2': 2.4994377999999995,
#                 'dm3': 2.4999333999999998,'dm4': 2.4999981,'dm5': 2.499663,'da1': 2.4991584,'da2': 2.4996680000000002,
#                 'da3': 2.4999561000000003,'da4': 2.4999662999999996,'da5': 2.4997887999999997,'db1': 2.4999776000000002,
#                 'db2': 2.4996115000000003,'db3': 2.4998112999999997,'db4': 2.4999968999999997,'db5': 2.4996541000000003}
#
# variables = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'thetad1',
#             'thetad2', 'thetad3', 'thetad4', 'thetad5', 'thetad6', 'tau1', 'tau2',
#             'tau3', 'tau4', 'tau5', 'dm1', 'dm2', 'dm3', 'dm4', 'dm5', 'da1', 'da2',
#             'da3', 'da4', 'da5', 'db1', 'db2', 'db3', 'db4', 'db5', 'thetadd6']
# model_name = '/pumadyn_32nm_ML.txt'


# -------->  GET SPARK CONTEXT <------------------
sc = SparkSession.builder.getOrCreate().sparkContext
spark = SparkSession(sc)
sqlContext = sql.SQLContext(sc)

# -------->  Read Ground Truth ML Model  <------------------
try:
    #input_file = r'/user/vs162304/Paper_AL/01_Data/PowerPlant_ML.txt'
    input_file = '/user/vs162304/Paper_AL/01_Data' + model_name
    file_object = open(input_file, 'r')
    str_mdl = file_object.read()
    reference_regr = pickle.loads(str_mdl.encode())
except:
    # input_file = r'/user/vs162304/Paper_AL/01_Data/PowerPlant_ML.txt'
    # #read string
    # f = sc.wholeTextFiles(input_file)
    # model_str = f.take(1)[0][1]  # .take(0)#(0)#._2
    # reference_regr = pickle.loads(model_str.encode())

    input_file = '../../01_Data' + model_name
    file_object = open(input_file, 'rb')
    str_mdl = file_object.read()
    reference_regr = pickle.loads(str_mdl)


# try:
#     #input_file = r'../../01_Data' + model_name
#     model_name = '/PowerPlant_ML.txt'
#     input_file = '../../01_Data' + model_name
#     file_object = open(input_file, 'r')
# except:
#     input_file = r'/user/vs162304/Paper_AL/01_Data/PowerPlant_ML.txt'
#     file_object = open(input_file, 'r')

# str_mdl = file_object.read()
# reference_regr = pickle.loads(str_mdl.encode())


# print('type is:' + str(type(f)))
# print('type f1' +str(type(f.take(1))))
# #print(f.values())
# print("extracted data")
# print('type f1' +str(type(f.take(1)[0][1])))
# print(f.take(1)[0][0])


# -------->  CREATION OF LISTS CONTAINING THE PREDICTIONS AND TARGETS  <------------------
preds_rr_mcs, targets_rr_mcs = [], []
preds_mlp_mcs, targets_mlp_mcs = [], []
preds_xgb_mcs, targets_xgb_mcs = [], []

preds_mcs, targets_mcs = {}, {}
preds_qbc, targets_qbc = {}, {}


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


# Generates random dataset for defined parameters in defined range
def random_generator(samples_, actual_lows, actual_highs, variables, model):
    # actual_lows = {'AT': [2], 'V': [30], 'AP': [993], 'RH': [30]}  # , 'PE':[425]}
    # actual_highs = {'AT': [35], 'V': [80], 'AP': [1033], 'RH': [100]}  # , 'PE':[495]}
    # variables = ['AT', 'V', 'AP', 'RH', 'PE']
    # samples_ = 100000
    #np.random.seed(5234)
    df_doe = pd.DataFrame(columns=variables[:-1])
    for var in variables[:-1]:
        df_doe[var] = np.random.uniform(actual_lows[var], actual_highs[var], samples_).round(3)
    df_doe[variables[-1]] = model.predict(df_doe).round(3)
    return df_doe


# Generates a LHC DOE design for defined parameters in defined range
def latin_hypercube_generator(samples_, actual_lows, actual_highs, variables, model):
    # actual_lows = {'AT': [2], 'V': [30], 'AP': [993], 'RH': [30]}  # , 'PE':[425]}
    # actual_highs = {'AT': [35], 'V': [80], 'AP': [1033], 'RH': [100]}  # , 'PE':[495]}
    # variables = ['AT', 'V', 'AP', 'RH', 'PE']
    # samples_ = 100000
    #np.random.seed(5234)
    df_doe = pd.DataFrame(lhs(len(variables) - 1, samples=samples_, criterion='maximin'))
    df_doe.columns = variables[:-1]
    for col in df_doe.columns:
        df_doe[col] = [actual_lows[col]] * df_doe.shape[0] + df_doe[col] * (actual_highs[col] - actual_lows[col])
        df_doe[col] = df_doe[col].apply(lambda x: round(x, 3))
    # df_doe = df_doe.reset_index()
    df_doe.columns = variables[:-1]
    df_doe[variables[-1]] = model.predict(df_doe).round(3)
    return df_doe


# -------->  MACHINE LEARNING MODELS (HYPERPARAMETERS, COMMITTEE FOR QBC) <------------------
# Ridge Regression RR
rr_params_v1 = {
    'alpha': 5, 'max_iter': 4, 'normalize': False, 'solver': 'lsqr', 'tol': 0.003}
mcs_rr_regr = Ridge(**rr_params_v1)
qbc_rr_regr = Ridge(**rr_params_v1)
# mcs_rr_regr = KNeighborsRegressor(n_neighbors=10, weights='uniform')
# qbc_rr_regr = KNeighborsRegressor(n_neighbors=10, weights='uniform')
# Multilayer Perceptron Neural Network MLP
mlp_params_v1 = {
    'solver': 'adam', 'hidden_layer_sizes': (60, 60), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 3000}
mcs_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_mlp_regr = MLPRegressor(**mlp_params_v1)

# # eXtreme Gradient Boosting XGB
# xgb_params_v1 = {
#     'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
#     'booster': 'gbtree', 'n_jobs': -1}
# mcs_xgb_regr = XGBRegressor(**xgb_params_v1)
# qbc_xgb_regr = XGBRegressor(**xgb_params_v1)
lgbmParams = {  # 'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'regression'}
# model = lightgbm.LGBMRegressor(**lgbmParams)
mcs_xgb_regr = lightgbm.LGBMRegressor(**lgbmParams)
qbc_xgb_regr = lightgbm.LGBMRegressor(**lgbmParams)

# -------->  Regressors to run  <------------------
predictors_list = [qbc_xgb_regr, qbc_rr_regr, qbc_mlp_regr]
# predictors_list = [qbc_rr_regr]

# --------> HOLDOUT TEST DATA FOR ALL METHODS<------------------
# generate test data set
np.random.seed(5234)
test_data = latin_hypercube_generator(size_test_set, actual_lows, actual_highs, variables, reference_regr)
test_data = np.array(test_data)

trainingSetSizes = np.arange(seedSize, size_train_set + seedSize, batchSize)

# -------->  Task splitting <------------------
task_batches_mcs = []
task_batches_mcs_extension = []
task_batches_al = []


# -------->  MONTE CARLO SAMPLING ACTIVE LEARNING (MCS) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in MCS <------------------
def mcs_for_reg(par_list):  # give all parametres as a list
    # -------->  Get parameters for the giben job <------------------
    # test_loop
    predictor = par_list[0]
    test_loop = par_list[1]
    test_data = par_list[2]
    size_train_set = par_list[3]
    actual_lows = par_list[4]
    actual_highs = par_list[5]
    variables = par_list[6]
    reference_regr = par_list[7]

    # Generate train data of the defined size
    np.random.seed(test_loop*100)
    train_data = np.array(random_generator(size_train_set, actual_lows, actual_highs, variables, reference_regr))

    # -------->  PREPARATION TEST SET <------------------
    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    preds_mcs[pred_key] = []
    targets_mcs[pred_key] = []

    X_train = train_data[:, :-1]
    X_test = test_data[:, :-1]
    y_train = train_data[:, -1].reshape((-1, 1))
    y_test = test_data[:, -1].reshape((-1, 1))

    # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
    mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(X_train)
    mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(y_train)

    # -------->  MODEL FITTING  <------------------
    predictor.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
    # -------->  PREDICTION <------------------
    mcs_pred_y_norm = predictor.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
    # -------->  DENORMALIZATION OF PREDICTED VALUES <------------------
    mcs_pred_y = mcs_pred_y_norm * mcs_std_y_train + mcs_mean_y_train

    return [pred_key, test_loop + 1, size_train_set, mcs_pred_y, np.ravel(y_test), 'Done']
    # return [pred_key, test_loop + 1, size_train_set, mcs_pred_y]


# -------->  MONTE CARLO SAMPLING ACTIVE LEARNING (MCS) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in MCS <------------------
def mcs_for_reg_extension(par_list):  # give all parametres as a list
    # -------->  Get parameters for the giben job <------------------
    # test_loop
    predictor = par_list[0]
    cv_itter = par_list[1]
    batchSize = par_list[2]
    seedSize = par_list[3]
    test_data = par_list[4]
    trainingSetSizes = par_list[5]
    actual_lows = par_list[6]
    actual_highs = par_list[7]
    variables = par_list[8]
    reference_regr = par_list[9]

    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]

    preds_mcs_ext = {}
    targets_mcs_ext = {}

    print('calculating MCS for ' + pred_key)

    # Generate initial train data of the defined size
    np.random.seed(cv_itter*100)
    train_data = np.array(random_generator(seedSize, actual_lows, actual_highs, variables, reference_regr))

    for idx, batch in enumerate(trainingSetSizes):
        print('Model: {}, number of test loop {}, batch size {}'.format(pred_key, cv_itter, batch))
        # --------> STANDARDIZATION OF DATA SETS (subtraction of the mean, division by standard deviation) <------------------
        mean_data, std_data, data_all_norm = normalize(np.concatenate((train_data, test_data), axis=0))

        X_train = train_data[:, :-1]
        X_test = test_data[:, :-1]
        y_train = train_data[:, -1].reshape((-1, 1))
        y_test = test_data[:, -1].reshape((-1, 1))

        # NORMALIZED DATA
        X_train_norm = data_all_norm[:train_data.shape[0], :-1]
        y_train_norm = data_all_norm[:train_data.shape[0], -1].reshape((-1, 1))
        X_test_norm = data_all_norm[train_data.shape[0]:, :-1]
        y_test_norm = data_all_norm[train_data.shape[0]:, -1].reshape((-1, 1))

        #  -------->  MODEL FITTING  <------------------
        predictor.fit(X_train_norm, y_train_norm.ravel())
        #  -------->  PREDICTION  <------------------
        mcs_y_norm = predictor.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        mcs_y = mcs_y_norm * std_data[-1].reshape((-1, 1)) + mean_data[-1].reshape((-1, 1))

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_mcs_ext[batch] = np.ravel(mcs_y)
        targets_mcs_ext[batch] = np.ravel(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break
        else:
            # -------->  PREPARING NEXT BATCH <------------------
            #batch_size = batch - X_train.shape[0]
            new_data = np.array(random_generator(batchSize, actual_lows, actual_highs, variables, reference_regr))
            train_data = np.concatenate((train_data, new_data), axis=0)

    # return [pred_key, cv_itter + 1, size_train_set, mcs_pred_y, np.ravel(y_test), 'Done']
    # return [pred_key, cv_itter + 1, size_train_set, mcs_pred_y]
    return [pred_key, cv_itter + 1, size_train_set, preds_mcs_ext, targets_mcs_ext, 'Done']


# -------->  LHC SAMPLING ACTIVE LEARNING (MCS) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in MCS <------------------
def lhc_for_reg(par_list):  # give all parametres as a list
    # -------->  Get parameters for the giben job <------------------
    # test_loop
    predictor = par_list[0]
    test_loop = par_list[1]
    test_data = par_list[2]
    size_train_set = par_list[3]
    actual_lows = par_list[4]
    actual_highs = par_list[5]
    variables = par_list[6]
    reference_regr = par_list[7]
    first_training = par_list[9]

    # Generate train data of the defined size
    if first_training:
        train_data = par_list[8]
    else:
        np.random.seed(test_loop*100)
        train_data = np.array(
            latin_hypercube_generator(size_train_set, actual_lows, actual_highs, variables, reference_regr))

    # -------->  PREPARATION TEST SET <------------------
    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    preds_mcs[pred_key] = []
    targets_mcs[pred_key] = []

    X_train = train_data[:, :-1]
    X_test = test_data[:, :-1]
    y_train = train_data[:, -1].reshape((-1, 1))
    y_test = test_data[:, -1].reshape((-1, 1))

    # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
    mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(X_train)
    mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(y_train)

    # -------->  MODEL FITTING  <------------------
    predictor.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
    # -------->  PREDICTION <------------------
    mcs_pred_y_norm = predictor.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
    # -------->  DENORMALIZATION OF PREDICTED VALUES <------------------
    mcs_pred_y = mcs_pred_y_norm * mcs_std_y_train + mcs_mean_y_train

    return [pred_key, test_loop + 1, size_train_set, mcs_pred_y, np.ravel(y_test), 'Done', train_data]
    # return [pred_key, size_train_set, mcs_pred_y]


# -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in QBC<------------------

# -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in QBC<------------------
def qbc_for_reg(par_list):  # give all parametres as a list
    # [predictor, cv_itter, batchSize, seedSize, test_data, numCommitteeMembers, trainingSetSizes, actual_lows,
    #  actual_highs, variables, reference_regr]

    # -------->  RETRIEVE TRAININGS PARAMETERS  <------------------
    predictor = par_list[0]
    cv_itter = par_list[1]
    batchSize = par_list[2]
    seedSize = par_list[3]
    test_data = par_list[4]
    numCommitteeMembers = par_list[5]
    trainingSetSizes = par_list[6]
    actual_lows = par_list[7]
    actual_highs = par_list[8]
    variables = par_list[9]
    reference_regr = par_list[10]


    # Generate train data of the defined size
    #train_data = np.array(random_generator(size_train_set, actual_lows, actual_highs, variables, reference_regr))
    train_data = np.array(random_generator(trainingSetSizes[-1]+1, actual_lows, actual_highs, variables, reference_regr))

    # --------> STANDARDIZATION OF DATA SETS (subtraction of the mean, division by standard deviation) <------------------
    mean_data, std_data, data_all_norm = normalize(np.concatenate((train_data, test_data), axis=0))

    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    preds_qbc = {}
    targets_qbc = {}
    print('calculating QBC for ' + pred_key)

    X_train = train_data[:seedSize, :-1]
    X_pool = train_data[seedSize:, :-1]
    y_train = train_data[:seedSize, -1].reshape((-1, 1))
    y_pool = train_data[seedSize:, -1].reshape((-1, 1))
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # NORMALIZED DATA
    X_train_norm = data_all_norm[:seedSize, :-1]
    y_train_norm = data_all_norm[:seedSize, -1].reshape((-1, 1))
    X_pool_norm = data_all_norm[seedSize:train_data.shape[0], :-1]
    y_pool_norm = data_all_norm[seedSize:train_data.shape[0], -1].reshape((-1, 1))
    X_test_norm = data_all_norm[train_data.shape[0]:, :-1]
    y_test_norm = data_all_norm[train_data.shape[0]:, -1].reshape((-1, 1))

    # INITIALISIERUNG
    startTime_batch = datetime.now()
    # --------> CREATION OF THE COMMITTEE) <------------------
    learner_list = []
    for i in range(numCommitteeMembers):
        learner_list.append(ActiveLearner(estimator=predictor,
                                          # estimator=sklearn.base.clone(predictor),
                                          X_training=X_train_norm,
                                          y_training=y_train_norm.ravel(),
                                          bootstrap_init=True))

    committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

    for idx, batch in enumerate(trainingSetSizes):
        print('Model: {}, number of test loop {}, batch size {}'.format(pred_key, cv_itter, batch))

        if committee_predictor:
            #  -------->  MODEL FITTING  <------------------
            committee.teach(X_train_norm, y_train_norm.ravel())
            #  -------->  PREDICTION  <------------------
            qbc_y_norm = committee.predict(X_test_norm)
        else:
            #  -------->  MODEL FITTING  <------------------
            predictor.fit(X_train_norm, y_train_norm.ravel())
            #  -------->  PREDICTION  <------------------
            qbc_y_norm = predictor.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        qbc_y = qbc_y_norm * std_data[-1].reshape((-1, 1)) + mean_data[-1].reshape((-1, 1))

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_qbc[batch] = np.ravel(qbc_y)
        targets_qbc[batch] = np.ravel(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break

        else:
            # -------->  PREPARING NEXT BATCH <------------------
            if X_pool.shape[0] < batchSize:
                batchSize = X_pool.shape[0]

            query_idx, query_instance = committee.query(X=X_pool_norm, n_instances=batchSize)
            X_train = np.append(X_train, X_pool[query_idx], axis=0)
            y_train = np.append(y_train, y_pool[query_idx], axis=0)
            X_train_norm = np.append(X_train_norm, X_pool_norm[query_idx], axis=0)
            y_train_norm = np.append(y_train_norm, y_pool_norm[query_idx], axis=0)

            committee.teach(X_pool_norm[query_idx],
                            y_pool_norm.ravel()[query_idx])  # train committee on newly added data

            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            X_pool_norm = np.delete(X_pool_norm, query_idx, axis=0)
            y_pool_norm = np.delete(y_pool_norm, query_idx, axis=0)

        print('   ' + str(datetime.now() - startTime_batch))

    # return [pred_key, cv_itter + 1, size_train_set, np.ravel(preds_qbc[pred_key]), np.ravel(targets_qbc[pred_key]), 'Done']
    return [pred_key, cv_itter + 1, size_train_set, preds_qbc, targets_qbc, 'Done']
    # return [pred_key, size_train_set, preds_qbc, qbc_y_com]

# -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in QBC<------------------
def qbc_for_reg_lhc(par_list):  # give all parametres as a list
    # [predictor, cv_itter, batchSize, seedSize, test_data, numCommitteeMembers, trainingSetSizes, actual_lows,
    #  actual_highs, variables, reference_regr]

    # -------->  RETRIEVE TRAININGS PARAMETERS  <------------------
    predictor = par_list[0]
    cv_itter = par_list[1]
    batchSize = par_list[2]
    seedSize = par_list[3]
    test_data = par_list[4]
    numCommitteeMembers = par_list[5]
    trainingSetSizes = par_list[6]
    actual_lows = par_list[7]
    actual_highs = par_list[8]
    variables = par_list[9]
    reference_regr = par_list[10]
    train_data = par_list[11]


    # Generate train data of the defined size
    np.random.seed(cv_itter*100)
    # train_data = np.array(
    #     latin_hypercube_generator(seedSize, actual_lows, actual_highs, variables, reference_regr))
    pool_data = np.array(
        latin_hypercube_generator(trainingSetSizes[-1]+1-seedSize, actual_lows, actual_highs, variables, reference_regr))

    # --------> STANDARDIZATION OF DATA SETS (subtraction of the mean, division by standard deviation) <------------------
    mean_data, std_data, data_all_norm = normalize(np.concatenate((train_data, pool_data, test_data), axis=0))

    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    preds_qbc = {}
    targets_qbc = {}
    print('calculating QBC for ' + pred_key)

    X_train = train_data[:, :-1]
    X_pool = pool_data[:, :-1]
    y_train = train_data[:, -1].reshape((-1, 1))
    y_pool = pool_data[:, -1].reshape((-1, 1))
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # NORMALIZED DATA
    X_train_norm = data_all_norm[:train_data.shape[0], :-1]
    y_train_norm = data_all_norm[:train_data.shape[0], -1].reshape((-1, 1))
    X_pool_norm = data_all_norm[train_data.shape[0]:train_data.shape[0]+pool_data.shape[0], :-1]
    y_pool_norm = data_all_norm[train_data.shape[0]:train_data.shape[0]+pool_data.shape[0], -1].reshape((-1, 1))
    X_test_norm = data_all_norm[train_data.shape[0]+pool_data.shape[0]:, :-1]
    y_test_norm = data_all_norm[train_data.shape[0]+pool_data.shape[0]:, -1].reshape((-1, 1))

    # INITIALISIERUNG
    startTime_batch = datetime.now()
    # --------> CREATION OF THE COMMITTEE) <------------------
    learner_list = []
    for i in range(numCommitteeMembers):
        learner_list.append(ActiveLearner(estimator=predictor,
                                          # estimator=sklearn.base.clone(predictor),
                                          X_training=X_train_norm,
                                          y_training=y_train_norm.ravel(),
                                          bootstrap_init=True))

    committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

    for idx, batch in enumerate(trainingSetSizes):
        print('Model: {}, number of test loop {}, batch size {}'.format(pred_key, cv_itter, batch))

        if committee_predictor:
            #  -------->  MODEL FITTING  <------------------
            committee.teach(X_train_norm, y_train_norm.ravel())
            #  -------->  PREDICTION  <------------------
            qbc_y_norm = committee.predict(X_test_norm)
        else:
            #  -------->  MODEL FITTING  <------------------
            predictor.fit(X_train_norm, y_train_norm.ravel())
            #  -------->  PREDICTION  <------------------
            qbc_y_norm = predictor.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        qbc_y = qbc_y_norm * std_data[-1].reshape((-1, 1)) + mean_data[-1].reshape((-1, 1))

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_qbc[batch] = np.ravel(qbc_y)
        targets_qbc[batch] = np.ravel(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break

        else:
            # -------->  PREPARING NEXT BATCH <------------------
            if X_pool.shape[0] < batchSize:
                batchSize = X_pool.shape[0]

            query_idx, query_instance = committee.query(X=X_pool_norm, n_instances=batchSize)
            X_train = np.append(X_train, X_pool[query_idx], axis=0)
            y_train = np.append(y_train, y_pool[query_idx], axis=0)
            X_train_norm = np.append(X_train_norm, X_pool_norm[query_idx], axis=0)
            y_train_norm = np.append(y_train_norm, y_pool_norm[query_idx], axis=0)

            committee.teach(X_pool_norm[query_idx],
                            y_pool_norm.ravel()[query_idx])  # train committee on newly added data

            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            X_pool_norm = np.delete(X_pool_norm, query_idx, axis=0)
            y_pool_norm = np.delete(y_pool_norm, query_idx, axis=0)

        print('   ' + str(datetime.now() - startTime_batch))
    return [pred_key, cv_itter + 1, size_train_set, preds_qbc, targets_qbc, 'Done', train_data]

if __name__ == "__main__":
    # -------->  RUN PARALLEL PROZESSING <------------------
    # -------->  Split MCS & LHC tasks based on model and validation runs to run in parallel <------------------
    #first training set for lcs and  qbc_lhc
    np.random.seed(5234)
    train_data = np.array(
        latin_hypercube_generator(size_train_set, actual_lows, actual_highs, variables, reference_regr))

    task_batches = []
    for predictor in predictors_list:
        for cv_itter in range(num_iters_test):
            first_train = True
            for size_train_set in trainingSetSizes:
                task_batches_mcs.append(
                    [predictor, cv_itter, test_data, size_train_set, actual_lows, actual_highs, variables, reference_regr, train_data, first_train])
                first_train = False

    # -------->  Split MCS_extended tasks based on model and validation runs to run in parallel <------------------
    for predictor in predictors_list:
        for cv_itter in range(num_iters_test):
            task_batches_mcs_extension.append(
                [predictor, cv_itter, batchSize, seedSize, test_data, trainingSetSizes, actual_lows,
                 actual_highs, variables, reference_regr])

    # -------->  Split AL tasks based on model and validation runs to run in parallel <------------------
    for predictor in predictors_list:
        for cv_itter in range(num_iters_test):
            task_batches_al.append(
                [predictor, cv_itter, batchSize, seedSize, test_data, numCommitteeMembers, trainingSetSizes, actual_lows,
                 actual_highs, variables, reference_regr, train_data])

    mcs_for_reg(task_batches_mcs[0])
    mcs_for_reg_extension(task_batches_mcs_extension[0])
    lhc_for_reg(task_batches_mcs[0])
    qbc_for_reg(task_batches_al[0])
    qbc_for_reg_lhc(task_batches_al[0])
    print('starting spark')
    # -------->    EXECUTE MCS JOB <------------------
    # print(test_data)
    f_mcs = lambda x: mcs_for_reg(x)
    results_mcs = sc.parallelize(task_batches_mcs).map(f_mcs).collect()
    print(results_mcs)

    f_mcs_extension = lambda x: mcs_for_reg_extension(x)
    results_mcs_extension = sc.parallelize(task_batches_mcs_extension).map(f_mcs_extension).collect()
    print(results_mcs_extension)
    # # -------->    SAVE MCS RESULTS TO HDFS <------------------
    # for preds in results_mcs:
    #     pred = sc.parallelize(preds[3])
    #     pred.coalesce(1).saveAsTextFile(
    #         r"/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_mcs_fold_" + str(preds[1]) + "_taining_size_" + str(preds[2]))
    #
    #     targ = sc.parallelize(preds[4])
    #     targ.coalesce(1).saveAsTextFile(
    #         r"/user/vs162304/Paper_AL/03_Results/targets_" + preds[0] + "_mcs_fold_" + str(preds[1]) + "_taining_size_" + str(preds[2]))


    # -------->    EXECUTE LHC JOB <------------------
    f_lhc = lambda x: lhc_for_reg(x)
    results_lhc = sc.parallelize(task_batches_mcs).map(f_lhc).collect()
    # print(results_lhc)
    # # -------->    SAVE LHC RESULTS TO HDFS <------------------
    # for preds in results_lhc:
    #     pred = sc.parallelize(preds[3])
    #     pred.coalesce(1).saveAsTextFile(
    #         r"/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_lhc_fold_" + str(preds[1]) + "_taining_size_" + str(preds[2]))
    #
    #     targ = sc.parallelize(preds[4])
    #     targ.coalesce(1).saveAsTextFile(
    #         r"/user/vs162304/Paper_AL/03_Results/targets_" + preds[0] + "_lhc_fold_" + str(preds[1]) + "_taining_size_" + str(preds[2]))


    # -------->    EXECUTE QBC JOB <------------------
    # print(test_data)
    # qbc_for_reg(task_batches_al[0])
    f_al = lambda x: qbc_for_reg(x)
    #results_al = sc.parallelize(task_batches_al).map(f_al).collect()
    # print(results_al)
    # # -------->    SAVE QBC RESULTS TO HDFS <------------------
    # for preds in results_al:
    #     for batch_sz in preds[3].keys():
    #         pred = sc.parallelize(preds[3][batch_sz])
    #         pred.coalesce(1).saveAsTextFile(
    #             r"/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_al_fold_" + str(
    #                 preds[1]) + "_taining_size_" + str(batch_sz))
    #
    #         targ = sc.parallelize(preds[4][batch_sz])
    #         targ.coalesce(1).saveAsTextFile(
    #             r"/user/vs162304/Paper_AL/03_Results/targets_" + preds[0] + "_al_fold_" + str(
    #                 preds[1]) + "_taining_size_" + str(batch_sz))
    f_al_lhc = lambda x: qbc_for_reg_lhc(x)
    results_al_lhc = sc.parallelize(task_batches_al).map(f_al_lhc).collect()

    endTime = datetime.now()
    td = endTime - startTime1
    print('running time: {} days, {} hours, {} mins'.format(td.days, td.seconds // 3600, td.seconds // 60 % 60))


    # try:
    #     pickle.dump(results_al, 'results_al.pkl')
    #     pickle.dump(results_lhc, 'results_lhc.pkl')
    #     pickle.dump(results_mcs, 'results_mcs.pkl')
    # except:
    #     print('save to pkl failed')


    # -------->    visualize results <------------------
    def res_in_df_al(res_lst):
        rows_list = []
        results = {}
        for job_res in res_lst:
            model = job_res[0]
            cv_fold = job_res[1]
            train_sizes = job_res[3].keys()
            for train_size in train_sizes:
                pred_vals = job_res[3][train_size]
                real_vals = job_res[4][train_size]
                rmse_val = rmse(real_vals, pred_vals)
                r2_val = r2_score(real_vals, pred_vals)
                results = {'Model': model, 'CV': cv_fold, 'Training_Size': train_size,
                           'RMSE': rmse_val, 'r2_val': r2_val}
                rows_list.append(results)
        res_df = pd.DataFrame(rows_list)  # columns=['Model', 'CV', 'Training_Size', 'Pred', 'Real'])
        return res_df


    def res_in_df_mcs_lhc(res_lst):
        rows_list = []
        results = {}
        for job_res in res_lst:
            model = job_res[0]
            cv_fold = job_res[1]
            train_size = job_res[2]
            pred_vals = job_res[3]
            real_vals = job_res[4]
            rmse_val = rmse(real_vals, pred_vals)
            r2_val = r2_score(real_vals, pred_vals)
            results = {'Model': model, 'CV': cv_fold, 'Training_Size': train_size,
                       'RMSE': rmse_val, 'r2_val': r2_val}
            rows_list.append(results)
        res_df = pd.DataFrame(rows_list)  # columns=['Model', 'CV', 'Training_Size', 'Pred', 'Real'])
        return res_df


    #results_al_rmse = res_in_df_al(results_al)
    results_al_lhc_rmse = res_in_df_al(results_al_lhc)
    results_mcs_extension_rmse = res_in_df_al(results_mcs_extension)
    #results_mcs_rmse = res_in_df_mcs_lhc(results_mcs)
    results_lhc_rmse = res_in_df_mcs_lhc(results_lhc)

    try:
        # create folder for results
        folder_name = model_name[1:-4] + 'SzTr_' + str(seedSize) + '+' + str(size_train_set) + '_SzTest_' + str(
            size_test_set) + '_CV_' + str(num_iters_test) + '_BatchSz_' + str(batchSize) + '_CommitteeSize_' + str(
            numCommitteeMembers)
        log_dir = Path.cwd().parent.parent / Path('03_Results') / Path(folder_name)
        log_dir.mkdir(parents=True, exist_ok=True)

        # results_al_rmse.to_pickle(str(log_dir) +str(Path("/")) + 'results_al.pkl')
        results_al_lhc_rmse.to_pickle(str(log_dir) + str(Path("/")) + 'results_al_lhc.pkl')
        # results_mcs_rmse.to_pickle(str(log_dir) + str(Path("/")) + 'results_mcs_rmse.pkl')
        results_mcs_extension_rmse.to_pickle(str(log_dir) + str(Path("/")) + 'results_mcs_extension.pkl')
        results_lhc_rmse.to_pickle(str(log_dir) + str(Path("/")) + 'results_lhc.pkl')
    except:
        print('save to pkl failed')

    # Plot
    import matplotlib.pyplot as plt


    def plot_learning_curve_rmse(df_target, label):
        df_target_filt = df_target[df_target.Model == mdl]
        df_target_mean = df_target_filt.groupby(['Model', 'Training_Size']).RMSE.mean().reset_index()
        df_target_std = df_target_filt.groupby(['Model', 'Training_Size']).RMSE.std().reset_index()
        plt.plot(df_target_mean.Training_Size, df_target_mean.RMSE, label=mdl + '_' + label, linestyle='-')
        plt.fill_between(df_target_mean.Training_Size, df_target_mean.RMSE - df_target_std.RMSE,
                         df_target_mean.RMSE + df_target_std.RMSE, alpha=0.2)
        ax.set_xlabel('Sample size')
        ax.set_ylabel('RMSE')
        ax.legend(loc='best')
        plt.title('Predictions')
        return plt


    def plot_learning_curve_r2(df_target, label):
        df_target_filt = df_target[df_target.Model == mdl]
        df_target_mean = df_target_filt.groupby(['Model', 'Training_Size']).r2_val.mean().reset_index()
        df_target_std = df_target_filt.groupby(['Model', 'Training_Size']).r2_val.std().reset_index()
        plt.plot(df_target_mean.Training_Size, df_target_mean.r2_val, label=mdl + '_' + label, linestyle='-')
        plt.fill_between(df_target_mean.Training_Size, df_target_mean.r2_val - df_target_std.r2_val,
                         df_target_mean.r2_val + df_target_std.r2_val, alpha=0.2)
        ax.set_xlabel('Sample size')
        ax.set_ylabel('r2')
        ax.legend(loc='best')
        plt.title('Predictions')
        return plt


    # color_dict = {'MLPRegressor': 'green'}
    for mdl in results_al_lhc_rmse.Model.unique():
        with plt.style.context('seaborn-white'):
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            # plot_learning_curve(results_mcs_rmse, 'mcs')
            plot_learning_curve_r2(results_mcs_extension_rmse, 'mcs_ext')
            plot_learning_curve_r2(results_lhc_rmse, 'lhc')
            # plot_learning_curve_r2(results_al_rmse, 'al')
            plot_learning_curve_r2(results_al_lhc_rmse, 'al_lhc')
            # fig.savefig('al_lhc_mcs_results' + mdl + '.png', dpi=300)
            plt.show()

    # for mdl in results_al_lhc_rmse.Model.unique():
    #     with plt.style.context('seaborn-white'):
    #         fig = plt.figure(figsize=(10, 5))
    #         ax = fig.add_subplot(111)
    #         # plot_learning_curve(results_mcs_rmse, 'mcs')
    #         plot_learning_curve_rmse(results_mcs_extension_rmse, 'mcs_ext')
    #         plot_learning_curve_rmse(results_lhc_rmse, 'lhc')
    #         # plot_learning_curve_rmse(results_al_rmse, 'al')
    #         plot_learning_curve_rmse(results_al_lhc_rmse, 'al_lhc')
    #         # fig.savefig('al_lhc_mcs_results' + mdl + '.png', dpi=300)
    #         plt.show()

