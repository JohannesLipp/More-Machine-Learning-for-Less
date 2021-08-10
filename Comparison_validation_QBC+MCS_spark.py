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
import subprocess
import lightgbm


def run_cmd(args_list):
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (output, errors) = proc.communicate()
    if proc.returncode:
        raise RuntimeError(
            'Error running command: %s. Return code: %d, Error: %s' % (
                ' '.join(args_list), proc.returncode, errors))


startTime1 = datetime.now()
# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

batchSize = 20  # size of the training set increments
seedSize = 10  # size of the initial training set
num_iters_test = 10  # choose desired number of cv iterations
size_test_set = 0.1  # choose desired size of the test set
numCommitteeMembers = 5  # choose number of committee members for the QBC

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


# -------->  GET SPARK CONTEXT <------------------
sc = SparkSession.builder.getOrCreate().sparkContext
spark = SparkSession(sc)
sqlContext = sql.SQLContext(sc)

# -------->  MACHINE LEARNING MODELS (HYPERPARAMETERS, COMMITTEE FOR QBC) <------------------

# Ridge Regression RR
rr_params_v1 = {
    'alpha': 5, 'max_iter': 4, 'normalize': False, 'solver': 'lsqr', 'tol': 0.003}
mcs_rr_regr = Ridge(**rr_params_v1)
qbc_rr_regr = Ridge(**rr_params_v1)

# Multilayer Perceptron Neural Network MLP
mlp_params_v1 = {
    'solver': 'adam', 'hidden_layer_sizes': (60, 60), 'activation': 'relu', 'tol': 1e-5, 'max_iter': 3000}
mcs_mlp_regr = MLPRegressor(**mlp_params_v1)
qbc_mlp_regr = MLPRegressor(**mlp_params_v1)

# eXtreme Gradient Boosting XGB
xgb_params_v1 = {
    'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
    'booster': 'gbtree', 'n_jobs': -1}
mcs_xgb_regr = XGBRegressor(**xgb_params_v1)
qbc_xgb_regr = XGBRegressor(**xgb_params_v1)


# -------->  Regressors to run  <------------------
#predictors_list = [qbc_xgb_regr, qbc_rr_regr, qbc_mlp_regr]
predictors_list = [qbc_rr_regr]

# -------->  DATA INPUT  <------------------
try:
    input_file = r'/user/vs162304/Paper_AL/01_Data/auto_mpg.csv'
    input_name = input_file[input_file.rfind('/') + 1: -4]  # mark results with inpud dataset name
    df_data = spark.read.format("csv").option("header", "true").load(input_file).toPandas()
    df_data = df_data.apply(pd.to_numeric, errors="ignore")
    df_data = df_data.fillna(value=0)
    arr_data = np.asarray(df_data)
except:
    input_file = r'../01_Data/auto_mpg.csv'
    df_data = pd.read_csv(input_file, header=0)
    df_data = df_data.fillna(value=0)
    arr_data = np.asarray(df_data)
print(arr_data)
trainingSetSizes = np.arange(seedSize, arr_data.shape[0] * (1 - size_test_set), batchSize)


# -------->  Task splitting <------------------
task_batches_mcs = []
task_batches_al = []


# -------->  MONTE CARLO SAMPLING ACTIVE LEARNING (MCS) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in MCS <------------------
def mcs_for_reg(par_list):  # give all parametres as a list
    # -------->  Get parameters for the giben job <------------------
    #test_loop
    predictor = par_list[0]
    test_loop = par_list[1]
    size_test_set = par_list[2]
    lc_loop = par_list[3]
    arr_data = par_list[4]

    # -------->  PREPARATION TEST SET <------------------
    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    preds_mcs[pred_key] = []
    targets_mcs[pred_key] = []

    np.random.seed(test_loop)
    pool = arr_data.copy()
    np.random.shuffle(pool)
    idx_split = int(pool.shape[0] * (1-size_test_set))
    X_train = pool[:idx_split, :-1]
    X_test = pool[idx_split:, :-1]
    y_train = pool[:idx_split, -1].reshape((-1, 1))
    y_test = pool[idx_split:, -1].reshape((-1, 1))


    # -------->  MONTE CARLO SAMPLING (MCS) <------------------
    mcs_X_train_split = X_train[:int(lc_loop), :]
    mcs_y_train_split = y_train[:int(lc_loop)]

    # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
    mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(mcs_X_train_split)
    mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(mcs_y_train_split)

    # -------->  MODEL FITTING  <------------------
    predictor.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
    # -------->  PREDICTION <------------------
    mcs_pred_y_norm = predictor.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
    # -------->  DENORMALIZATION OF PREDICTED VALUES <------------------
    mcs_pred_y = mcs_pred_y_norm * mcs_std_y_train + mcs_mean_y_train

    return [pred_key, test_loop + 1, lc_loop, mcs_pred_y, np.ravel(y_test), 'Done']

# -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs  <------------------
# -------->  Create a function that would be used as a lambda function for parallel processing in QBC<------------------
def qbc_for_reg(par_list):  # give all parametres as a list

    # -------->  RETRIEVE TRAININGS PARAMETERS  <------------------
    predictor = par_list[0]
    cv_itter = par_list[1]
    batchSize = par_list[2]
    seedSize = par_list[3]
    size_test_set = par_list[4]
    numCommitteeMembers = par_list[5]
    trainingSetSizes = par_list[6]
    arr_data = par_list[7]

    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    preds_qbc[pred_key] = []
    targets_qbc[pred_key] = []
    print('calculating QBC for ' + pred_key)

    np.random.seed(cv_itter)
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
        print('Model: {}, number of test loop {}, batch size {}'.format(pred_key, cv_itter, batch))

        # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
        qbc_mean_X_train, qbc_std_X_train, qbc_X_train_split_norm = normalize(qbc_X_train_split)
        qbc_mean_y_train, qbc_std_y_train, qbc_y_train_split_norm = normalize(qbc_y_train_split)
        qbc_X_test_split_norm = (qbc_X_test_split - qbc_mean_X_train) / qbc_std_X_train
        qbc_y_test_split_norm = (qbc_y_test_split - qbc_mean_y_train) / qbc_std_y_train

        X_test_norm = (X_test - qbc_mean_X_train) / qbc_std_X_train
        y_test_norm = (y_test - qbc_mean_y_train) / qbc_std_y_train

        # -------->  MODEL FITTING  <------------------
        predictor.fit(qbc_X_train_split_norm, qbc_y_train_split_norm.ravel())

        # -------->  PREDICTION  <------------------
        qbc_y_norm = predictor.predict(X_test_norm)

        # -------->  DENORMALIZATION OF PREDICTED VALUES  <------------------
        qbc_y = qbc_y_norm * qbc_std_y_train + qbc_mean_y_train

        # -------->  SAVING THE PREDICTED AND TARGET VALUES  <------------------
        preds_qbc[pred_key].append(qbc_y)
        targets_qbc[pred_key].append(y_test)

        if (idx + 1) == len(trainingSetSizes):
            break

        else:
            # --------> CREATION OF THE COMMITTEE) <------------------
            learner_list = []
            for i in range(numCommitteeMembers):
                learner_list.append(ActiveLearner(estimator=predictor,
                                                  # estimator=sklearn.base.clone(predictor),
                                                  X_training=qbc_X_train_split_norm,
                                                  y_training=qbc_y_train_split_norm.ravel(),
                                                  bootstrap_init=True))

            committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

            # -------->  PREPARING NEXT BATCH <------------------
            if qbc_X_test_split_norm.shape[0]<batchSize:
                batchSize = qbc_X_test_split_norm.shape[0]

            query_idx, query_instance = committee.query(X=qbc_X_test_split_norm, n_instances=batchSize)
            qbc_X_train_split = np.append(qbc_X_train_split, qbc_X_test_split[query_idx], axis=0)
            qbc_y_train_split = np.append(qbc_y_train_split, qbc_y_test_split[query_idx], axis=0)
            qbc_X_train_split_norm = np.append(qbc_X_train_split_norm, qbc_X_test_split_norm[query_idx], axis=0)
            qbc_y_train_split_norm = np.append(qbc_y_train_split_norm, qbc_y_test_split_norm[query_idx], axis=0)

            qbc_X_test_split = np.delete(qbc_X_test_split, query_idx, axis=0)
            qbc_y_test_split = np.delete(qbc_y_test_split, query_idx, axis=0)
            qbc_X_test_split_norm = np.delete(qbc_X_test_split_norm, query_idx, axis=0)
            qbc_y_test_split_norm = np.delete(qbc_y_test_split_norm, query_idx, axis=0)

            # for query in range(batchSize):
            #     # sample selection and committee training
            #     query_idx, query_instance = committee.query(qbc_X_test_split_norm)
            #     a = datetime.now()
            #     committee.teach(qbc_X_test_split_norm[query_idx], qbc_y_test_split_norm.ravel()[query_idx],
            #                     bootstrap=True)
            #     print('   ' + str(datetime.now() - a))
            #     # committee.rebag() #could be nt required, is already imlemented in teach with bootstrap=Ture
            #
            #     qbc_X_train_split = np.append(qbc_X_train_split, qbc_X_test_split[query_idx], axis=0)
            #     qbc_y_train_split = np.append(qbc_y_train_split, qbc_y_test_split[query_idx], axis=0)
            #     qbc_X_train_split_norm = np.append(qbc_X_train_split_norm, qbc_X_test_split_norm[query_idx], axis=0)
            #     qbc_y_train_split_norm = np.append(qbc_y_train_split_norm, qbc_y_test_split_norm[query_idx], axis=0)
            #
            #     qbc_X_test_split = np.delete(qbc_X_test_split, query_idx, axis=0)
            #     qbc_y_test_split = np.delete(qbc_y_test_split, query_idx, axis=0)
            #     qbc_X_test_split_norm = np.delete(qbc_X_test_split_norm, query_idx, axis=0)
            #     qbc_y_test_split_norm = np.delete(qbc_y_test_split_norm, query_idx, axis=0)

        print('   ' + str(datetime.now() - startTime_batch))
    # print('Test loop run time: ' + str(datetime.now() - startTime))
    # np.savetxt(r"/user/vs162304/Paper_AL/03_Results/preds_" + pred_key + "_qbc_fold" + str(cv_itter+1) + ".csv", np.ravel(preds_qbc[pred_key]), delimiter=",")
    # np.savetxt(r"/user/vs162304/Paper_AL/03_Results/targets_" + pred_key + "_qbc_fold" + str(cv_itter+1) + ".csv", np.ravel(targets_qbc[pred_key]), delimiter=",")

    return [pred_key, cv_itter + 1, np.ravel(preds_qbc[pred_key]), np.ravel(targets_qbc[pred_key]), 'Done']


# -------->  RUN PARALLEL PROZESSING <------------------
# -------->  Split MCS tasks based on model and validation runs and to run in parallel <------------------
task_batches = []
for predictor in predictors_list:
    for cv_itter in range(num_iters_test):
        for lc_loop in trainingSetSizes:
            task_batches_mcs.append([predictor, cv_itter, size_test_set, lc_loop, arr_data])

# -------->  Split AL tasks based on model and validation runs to run in parallel <------------------
for predictor in predictors_list:
    for cv_itter in range(num_iters_test):
        task_batches_al.append(
            [predictor, cv_itter, batchSize, seedSize, size_test_set, numCommitteeMembers, trainingSetSizes, arr_data])

f_mcs = lambda x: mcs_for_reg(x)
results_mcs = sc.parallelize(task_batches_mcs).map(f_mcs).collect()
print(results_mcs)

f_al = lambda x: qbc_for_reg(x)
results_al = sc.parallelize(task_batches_al).map(f_al).collect()
print(results_al)

# -------->    SAVE RESULTS TO HDFS <------------------
#MCS
for preds in results_al:
    pred_name = "/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_mcs_fold_" + str(preds[1])
    try:
        cmd = ('hadoop fs -rm -R' + pred_name).split()
        (out, errors) = run_cmd(cmd)
    except:
        a = 1
    pred = sc.parallelize(preds[2])
    # rdd_pred.saveAsTextFile("/user/vs162304/Paper_AL/03_Results/preds_" + pred_key + "_qbc_fold" + str(cv_itter+1) + ".csv")
    # pred.coalesce(1).write.format('com.databricks.spark.csv').options(header='false').save(r"/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_qbc_fold" + str(preds[1]))
    pred.coalesce(1).saveAsTextFile(
        r"/user/vs162304/Paper_AL/03_Results/" + input_name + "_preds_" + preds[0] + "_qbc_fold_" + str(preds[1]) + "_taining_size_" + str(preds[1]))

    target_name = "/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_qbc_fold" + str(preds[1])
    try:
        cmd = ('hadoop fs -rm -R' + target_name).split()
        run_cmd(cmd)
    except:
        a = 1
    targ = sc.parallelize(preds[3])
    targ.coalesce(1).saveAsTextFile(
        r"/user/vs162304/Paper_AL/03_Results/" + input_name + "_targets_" + preds[0] + "_qbc_fold" + str(preds[1]))

#AL
for preds in results_al:
    pred_name = "/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_qbc_fold" + str(preds[1])
    try:
        cmd = ('hadoop fs -rm -R' + pred_name).split()
        (out, errors) = run_cmd(cmd)
    except:
        a = 1
    pred = sc.parallelize(preds[2])
    # rdd_pred.saveAsTextFile("/user/vs162304/Paper_AL/03_Results/preds_" + pred_key + "_qbc_fold" + str(cv_itter+1) + ".csv")
    # pred.coalesce(1).write.format('com.databricks.spark.csv').options(header='false').save(r"/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_qbc_fold" + str(preds[1]))
    pred.coalesce(1).saveAsTextFile(
        r"/user/vs162304/Paper_AL/03_Results/" + input_name + "_preds_" + preds[0] + "_qbc_fold" + str(preds[1]))

    target_name = "/user/vs162304/Paper_AL/03_Results/preds_" + preds[0] + "_qbc_fold" + str(preds[1])
    try:
        cmd = ('hadoop fs -rm -R' + target_name).split()
        run_cmd(cmd)
    except:
        a = 1
    targ = sc.parallelize(preds[3])
    targ.coalesce(1).saveAsTextFile(
        r"/user/vs162304/Paper_AL/03_Results/" + input_name + "_targets_" + preds[0] + "_qbc_fold" + str(preds[1]))

