import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from modAL.disagreement import max_std_sampling
from datetime import datetime
#import tensorflow as tf

startTime1 = datetime.now()


# class Wrap_NN_A():
#
#     def __init__(self, regularize=2e-6):
#         tf.reset_default_graph()
#         self.regularize = regularize
#         # start tensorflow session
#         self.sess = tf.Session()
#
#         # create network
#         with tf.variable_scope("Reg_NN_Acceleration"):
#             self.data_x = tf.placeholder(tf.float32, [None, 7], 'Input_X')
#             self.data_y = tf.placeholder(tf.float32, [None, 1], 'Input_Y')
#
#             self.weights_reg = tf.contrib.layers.l2_regularizer(self.regularize)
#             self.layer1 = tf.layers.dense(self.data_x, 60, activation=tf.nn.relu, kernel_regularizer=self.weights_reg,
#                                           name='layer1')
#             # self.layer2 = tf.layers.dense(self.layer1, 60, activation=tf.nn.relu, kernel_regularizer=self.weights_reg,
#             #                               name='layer2')
#             # self.layer3 = tf.layers.dense(self.layer2, 300, activation=tf.nn.relu, kernel_regularizer=self.weights_reg,
#             #                               name='layer3')
#             self.layer2 = tf.layers.dense(self.layer1, 60, activation=tf.nn.relu, kernel_regularizer=self.weights_reg,
#                                           name="layer4")
#             self.out = tf.layers.dense(self.layer2, 1, activation=tf.nn.sigmoid, kernel_regularizer=self.weights_reg,
#                                        name='out_a')
#             self.loss = tf.losses.mean_squared_error(labels=self.data_y,
#                                                      predictions=self.out) + tf.losses.get_regularization_loss()
#             self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
#
#     def fit(self, X_train, y_train):
#         y_train = y_train.reshape((-1,1))
#         # start tensorflow session
#         self.sess.run(tf.global_variables_initializer())
#
#         # train network
#         for i in range(num_iteration):
#             index = np.random.randint(0, len(X_train), size=32)
#             batch_X = X_train[index]
#             #batch_X = np.swapaxes(batch_X, 1, 0)
#             batch_y = y_train[index]#.reshape((-1,1))
#             self.sess.run(self.train_op, feed_dict={self.data_x: batch_X, self.data_y: batch_y})
#             # if i % 100 == 0:
#             #     loss_train_ = self.sess.run(self.loss, feed_dict={self.data_x: X_train, self.data_y: y_train})
#             #     print('Step: {}\tLoss_train: {}'.format(i, loss_train_))
#
#     def predict(self, X_test):
#
#         # test network
#         prediction = self.sess.run(self.out, feed_dict={self.data_x: X_test})
#
#         return prediction

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

batchSize = 20             # size of the training set increments
seedSize = 10              # size of the initial training set
num_iters_test = 5       # choose desired number of test loop iterations
size_test_set = 0.1        # choose desired size of the test set
numCommitteeMembers = 5    # choose number of committee members for the QBC


# -------->  CREATION OF LISTS CONTAINING THE PREDICTIONS AND TARGETS  <------------------
preds_rr_mcs, targets_rr_mcs = [], []
preds_mlp_mcs, targets_mlp_mcs = [], []
preds_xgb_mcs, targets_xgb_mcs = [], []

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

# # Tensorflow NN Regressor
# NN_reg = Wrap_NN_A()

# predictors_list = [qbc_xgb_regr, qbc_rr_regr, qbc_mlp_regr]
# rr_params_v1 = {
#     'alpha': 5, 'max_iter': 4, 'normalize': False, 'solver': 'lsqr', 'tol': 0.003}
# rr_params_v2 = {
#     'alpha': 8, 'max_iter': 10, 'normalize': False, 'solver': 'lsqr', 'tol': 0.0001}
# qbc_rr_regr1 = Ridge(**rr_params_v1)
# qbc_rr_regr2 = Ridge(**rr_params_v2)
# #predictors_list = [qbc_rr_regr1, qbc_rr_regr2]
xgb_params_v1 = {
    'max_depth': 2, 'learning_rate': 0.3, 'n_estimators': 1250, 'silent': 1, 'eta': 0.3, 'min_child_weight': 5,
    'booster': 'gbtree', 'n_jobs': 1}
mcs_xgb_regr = XGBRegressor(**xgb_params_v1)
predictors_list = [qbc_rr_regr]

# -------->  DATA INPUT  <------------------
df_data = pd.read_csv(
    r'../01_Data/auto_mpg.csv', header=0)
df_data = df_data.fillna(value=0)
arr_data = np.asarray(df_data)
trainingSetSizes = np.arange(seedSize, arr_data.shape[0] * (1-size_test_set), batchSize)


# # -------->  MONTE CARLO SAMPLING ACTIVE LEARNING (MCS) DOEs  <------------------
# print('calculating MCS...')
#
# # -------->  ITERATIONS OVER DIFFERENT TEST SETS: TEST LOOP  <------------------
#
# for test_loop in range(num_iters_test):
#     startTime = datetime.now()
#
#     # -------->  PREPARATION TEST SET <------------------
#     np.random.seed(test_loop)
#     pool = arr_data.copy()
#     np.random.shuffle(pool)
#     idx_split = int(pool.shape[0] * (1-size_test_set))
#     X_train = pool[:idx_split, :-1]
#     X_test = pool[idx_split:, :-1]
#     y_train = pool[:idx_split, -1].reshape((-1, 1))
#     y_test = pool[idx_split:, -1].reshape((-1, 1))
#
#
#     for lc_loop in range(len(trainingSetSizes)):
#         startTime_batch = datetime.now()
#         print(str(test_loop) + '.' + str(lc_loop))
#
#         # -------->  MONTE CARLO SAMPLING (MCS) <------------------
#         mcs_X_train_split = X_train[:int(trainingSetSizes[lc_loop]), :]
#         mcs_y_train_split = y_train[:int(trainingSetSizes[lc_loop])]
#
#         # --------> STANDARDIZATION OF DATA SET (subtraction of the mean, division by standard deviation) <------------------
#         mcs_mean_X_train, mcs_std_X_train, mcs_X_train_split_norm = normalize(mcs_X_train_split)
#         mcs_mean_y_train, mcs_std_y_train, mcs_y_train_split_norm = normalize(mcs_y_train_split)
#
#         # -------->  MODEL FITTING  <------------------
#         mcs_mlp_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
#         mcs_xgb_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
#         mcs_rr_regr.fit(mcs_X_train_split_norm, mcs_y_train_split_norm.ravel())
#
#         # -------->  PREDICTION <------------------
#         mcs_mlp_y_norm = mcs_mlp_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
#         mcs_xgb_y_norm = mcs_xgb_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
#         mcs_rr_y_norm = mcs_rr_regr.predict((X_test - mcs_mean_X_train) / mcs_std_X_train)
#
#         # -------->  DENORMALIZATION OF PREDICTED VALUES <------------------
#         mcs_mlp_y = mcs_mlp_y_norm * mcs_std_y_train + mcs_mean_y_train
#         mcs_xgb_y = mcs_xgb_y_norm * mcs_std_y_train + mcs_mean_y_train
#         mcs_rr_y = mcs_rr_y_norm * mcs_std_y_train + mcs_mean_y_train
#
#         # -------->  SAVING PREDICTED VALUES <------------------
#         preds_rr_mcs.append(mcs_rr_y)
#         preds_mlp_mcs.append(mcs_mlp_y)
#         preds_xgb_mcs.append(mcs_xgb_y)
#         targets_rr_mcs.append(y_test)
#         targets_mlp_mcs.append(y_test)
#         targets_xgb_mcs.append(y_test)
#
#         print('   ' + str(datetime.now() - startTime_batch))
#     print('Test loop run time: ' + str(datetime.now() - startTime))




# -------->  QUERY BY COMMITTEE ACTIVE LEARNING (QBC) DOEs #each model been trained with comitee consisting of the same model kind <------------------
pred_names_list = []
for predictor in predictors_list:
    pred_str = str(predictor)
    pred_key = pred_str[:pred_str.find('(')]
    pred_names_list.append(pred_key)
    preds_qbc[pred_key] = []
    targets_qbc[pred_key] = []
    print('calculating QBC for ' + pred_key)


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
            print('Model: {}, number of test loop {}, batch size {}'.format(pred_key, test_loop, batch))

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
                                                      #estimator=sklearn.base.clone(predictor),
                                                      X_training=qbc_X_train_split_norm,
                                                      y_training=qbc_y_train_split_norm.ravel(),
                                                      bootstrap_init=True))

                committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

                # -------->  PREPARING NEXT BATCH <------------------
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
                #     a=datetime.now()
                #     committee.teach(qbc_X_test_split_norm[query_idx], qbc_y_test_split_norm.ravel()[query_idx], bootstrap=True)
                #     print('   ' + str(datetime.now() - a))
                #     #committee.rebag() #could be nt required, is already imlemented in teach with bootstrap=Ture
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
        print('Test loop run time: ' + str(datetime.now() - startTime))


# -------->  SAVING THE RESULTS  <------------------
print("Saving...")

np.savetxt("preds_rr_mcs.csv", np.ravel(preds_rr_mcs), delimiter=",")
np.savetxt("preds_mlp_mcs.csv", np.ravel(preds_mlp_mcs), delimiter=",")
np.savetxt("preds_xgb_mcs.csv", np.ravel(preds_xgb_mcs), delimiter=",")

np.savetxt("targets_rr_mcs.csv", np.ravel(targets_rr_mcs), delimiter=",")
np.savetxt("targets_mlp_mcs.csv", np.ravel(targets_mlp_mcs), delimiter=",")
np.savetxt("targets_xgb_mcs.csv", np.ravel(targets_xgb_mcs), delimiter=",")

for pred_key in pred_names_list:
    np.savetxt("preds_" + pred_key + "_qbc.csv", np.ravel(preds_qbc[pred_key]), delimiter=",")
    np.savetxt("targets_" + pred_key + "_qbc.csv", np.ravel(targets_qbc[pred_key]), delimiter=",")

print("Done. Running time: {}".format(datetime.now() - startTime1))