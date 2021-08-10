import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

batchSize = 10             # size of the training set increments
seedSize = 10              # size of the initial training set
num_iters_test = 20        # choose desired number of test loop iterations
size_test_set = 0.1        # choose desired size of the test set


df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/MLPlatform/data/auto_mpg.csv', header=0)

arr_data = np.asarray(df_data)

trainingSetSizes = np.arange(seedSize, arr_data.shape[0] * (1-size_test_set), batchSize)
idx_split = int(arr_data.shape[0] * (1-size_test_set))
testSetSize = int(arr_data[idx_split:, :].shape[0])
lenTestcycle = testSetSize * len(trainingSetSizes)


# -------->  FUNCTIONS  <------------------
# RMSE CALCULATION
def rmse(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2.
    return np.sqrt(residuals / n)

# -------->  DATA IMPORT  <------------------
print('Importing files...')


# data
df_rr_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/preds_rr_mcs.csv', header=None)
df_mlp_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/preds_mlp_mcs.csv', header=None)
df_xgb_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/preds_xgb_mcs.csv', header=None)
df_rr_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/preds_rr_qbc.csv', header=None)
df_mlp_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/preds_mlp_qbc.csv', header=None)
df_xgb_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/preds_xgb_qbc.csv', header=None)

df_targets_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/targets_mlp_mcs.csv', header=None)
df_targets_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/targets_mlp_qbc.csv', header=None)

df_mcs = pd.concat([df_rr_mcs, df_mlp_mcs, df_xgb_mcs], axis=1)
df_qbc = pd.concat([df_rr_qbc, df_mlp_qbc, df_xgb_qbc], axis=1)

print('Converting to np-arrays...')
preds_mcs = np.asarray(df_mcs)
preds_qbc = np.asarray(df_qbc)
targets_mcs = np.asarray(df_targets_mcs)
targets_qbc = np.asarray(df_targets_qbc)

def calculator_rmse_std_r2(preds_mcs, preds_qbc, targets_mcs, targets_qbc):
    results_table = np.zeros([1+3*3*3, int(len(trainingSetSizes) + 2)])

    # -------->  INPUT TRAININGSET SIZES  <------------------
    for i in range(len(trainingSetSizes)):
        results_table[0, i] = trainingSetSizes[i]

    # -------->  RMSE  <------------------
    # -------->  MCS  <------------------
    print(targets_mcs.shape[0], lenTestcycle * num_iters_test, lenTestcycle, num_iters_test)

    assert (targets_mcs.shape[0] == lenTestcycle * num_iters_test)

    for idx in range(len(trainingSetSizes)):

        for model in range(3):
            list_rmse = []

            for test in range(num_iters_test):

                test_loop = test * lenTestcycle
                counter = idx * testSetSize

                start_idx = test_loop + counter
                end_idx = start_idx + testSetSize

                rmse_value = rmse(targets_mcs[start_idx: end_idx],
                                  preds_mcs[start_idx: end_idx, model])

                print('MCS split ' + str(idx + 1) + ', test ' + str(test + 1) + ', model ' + str(model + 1) + ', preds '+
                      str(start_idx) + ' - ' + str(end_idx) + ', testSetSize: ' + str(len(targets_mcs[start_idx: end_idx]))
                      + ', RMSE: ' + str(np.round(rmse_value, decimals=2)))

                print(' ')
                list_rmse.append(rmse_value)


            results_table[model + 1, idx] = np.mean(list_rmse)


    # -------->  QBC  <------------------
    print(targets_qbc.shape[0], lenTestcycle * num_iters_test)

    assert (targets_qbc.shape[0] == lenTestcycle * num_iters_test)

    for idx in range(len(trainingSetSizes)):

        for model in range(3):
            list_rmse = []

            for test in range(num_iters_test):
                test_loop = test * lenTestcycle
                counter = idx * testSetSize

                start_idx = test_loop + counter
                end_idx = start_idx + testSetSize

                rmse_value = rmse(targets_qbc[start_idx: end_idx],
                                  preds_qbc[start_idx: end_idx, model])

                print('QBC split ' + str(idx + 1) + ', test ' + str(test + 1) + ', model ' + str(model + 1) + ', preds ' +
                      str(start_idx) + ' - ' + str(end_idx) + ', testSetSize: ' + str(
                    len(targets_qbc[start_idx: end_idx]))
                      + ', RMSE: ' + str(np.round(rmse_value, decimals=2)))

                print(' ')
                list_rmse.append(rmse_value)

            results_table[model + 7, idx] = np.mean(list_rmse)

    return results_table



print('Calculating...')

table = calculator_rmse_std_r2(preds_mcs=preds_mcs, preds_qbc=preds_qbc,
                               targets_mcs=targets_mcs, targets_qbc=targets_qbc)


print('Saving...')

pd.DataFrame(table[1:, :], columns=[table[0]]).to_csv('results.csv')
