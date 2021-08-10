import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# copy the settings used in PN_QBC+MCS_JO_data.py !!!
num_iters_train = 5
num_iters_test = 5
batchSize = 10

# same dataset as in PN_QBC+MCS_JO_data.py has to be used
df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/MLPlatform/data/auto_mpg.csv', header=0)

arr_data = np.asarray(df_data)
frac = batchSize / arr_data.shape[0]
splits = np.arange(frac, 1, frac)
sams = splits * arr_data.shape[0]

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



# data import are the .csv outputs from PN_QBC+MCS_JO_data.py
df_rr_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/preds_rr_mcs.csv', header=None)
df_mlp_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/preds_mlp_mcs.csv', header=None)
df_xgb_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/preds_xgb_mcs.csv', header=None)

df_rr_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/preds_rr_qbc.csv', header=None)
df_mlp_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/preds_mlp_qbc.csv', header=None)
df_xgb_qbc = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/preds_xgb_qbc.csv', header=None)


df_targets_mcs = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-12/targets_mlp_mcs.csv', header=None)

df_mcs = pd.concat([df_rr_mcs, df_mlp_mcs, df_xgb_mcs], axis=1)
df_qbc = pd.concat([df_rr_qbc, df_mlp_qbc, df_xgb_qbc], axis=1)

print('Converting to np-arrays...')
preds_mcs = np.asarray(df_mcs)
preds_qbc = np.asarray(df_qbc)
targets_mcs = np.asarray(df_targets_mcs)



def calculator_rmse_mcsqbc_r2(mcs, qbc, targets_mcs):
    testsetsize = 40    # manual input! should always be equal to np.ceil(arr_data.shape[0]*0.1)
    results_table = np.zeros([1+3*3*3, int(len(splits) + 2)])   # creates final output table

    # -------->  INPUT SPLITS  <------------------
    for i in range(len(splits)):
        results_table[0, i] = splits[i]

    # -------->  RMSE, STD of RMSEs  <------------------
    # -------->  MCS  <------------------
    print(len(splits))
    print(targets_mcs.shape, testsetsize*len(splits)*num_iters_train*num_iters_test, testsetsize, len(splits), num_iters_test, num_iters_train)
    assert(targets_mcs.shape[0] == testsetsize*len(splits)*num_iters_train*num_iters_test)
    for split in range(len(splits)):
        for model in range(3):
            list_rmse = []
            for test in range(num_iters_test):
                for train in range(num_iters_train):
                    cycle = test * num_iters_train * len(splits) * testsetsize
                    counter = train * len(splits) * testsetsize
                    rmse_value = rmse(targets_mcs[(split*testsetsize + cycle + counter): (split*testsetsize + cycle + counter + testsetsize)],
                                                  mcs[split*testsetsize + cycle + counter: split*testsetsize + cycle + counter + testsetsize, model])
                    print(split, model, np.round(rmse_value*1000, decimals=2), (split * testsetsize + cycle + counter),
                          (split * testsetsize + cycle + counter + testsetsize))
                    list_rmse.append(rmse_value)

            results_table[model + 1, split] = np.mean(list_rmse)
            results_table[model + 10, split] = np.std(list_rmse)


    # -------->  QBC  <------------------
    for split in range(len(splits)):
        for model in range(3):
            list_rmse = []
            for test in range(num_iters_test):
                for train in range(num_iters_train):
                    cycle = test * num_iters_train * len(splits) * testsetsize
                    counter = train * len(splits) * testsetsize
                    rmse_value = rmse(targets_mcs[(split*testsetsize + cycle + counter): (split*testsetsize + cycle + counter + testsetsize)],
                                                  qbc[split*testsetsize + cycle + counter: split*testsetsize + cycle + counter + testsetsize, model])
                    print(split, model, np.round(rmse_value*1000, decimals=2), (split * testsetsize + cycle + counter),
                          (split * testsetsize + cycle + counter + testsetsize))
                    list_rmse.append(rmse_value)

            results_table[model + 7, split] = np.mean(list_rmse)
            results_table[model + 16, split] = np.std(list_rmse)

    # -------->  R2  <------------------
    # -------->  MCS  <------------------
    print(len(splits))
    print(targets_mcs.shape, testsetsize * len(splits) * num_iters_train * num_iters_test, testsetsize, len(splits),
          num_iters_test, num_iters_train)
    assert (targets_mcs.shape[0] == testsetsize * len(splits) * num_iters_train * num_iters_test)
    for split in range(len(splits)):
        for model in range(3):
            list_r2 = []
            for test in range(num_iters_test):
                for train in range(num_iters_train):
                    cycle = test * num_iters_train * len(splits) * testsetsize
                    counter = train * len(splits) * testsetsize
                    r2_value = r2_score(targets_mcs[(split * testsetsize + cycle + counter): (split * testsetsize + cycle + counter + testsetsize)],
                                        mcs[split * testsetsize + cycle + counter: split * testsetsize + cycle + counter + testsetsize, model])
                    print(split, model, np.round(r2_value * 1000, decimals=2),
                          (split * testsetsize + cycle + counter),
                          (split * testsetsize + cycle + counter + testsetsize))
                    list_r2.append(r2_value)

            results_table[model + 19, split] = np.mean(list_r2)

    # -------->  QBC  <------------------
    for split in range(len(splits)):
        for model in range(3):
            list_r2 = []
            for test in range(num_iters_test):
                for train in range(num_iters_train):
                    cycle = test * num_iters_train * len(splits) * testsetsize
                    counter = train * len(splits) * testsetsize
                    r2_value = r2_score(targets_mcs[(split * testsetsize + cycle + counter): (split * testsetsize + cycle + counter + testsetsize)],
                                      qbc[split * testsetsize + cycle + counter: split * testsetsize + cycle + counter + testsetsize, model])
                    print(split, model, np.round(r2_value * 1000, decimals=2),
                          (split * testsetsize + cycle + counter),
                          (split * testsetsize + cycle + counter + testsetsize))
                    list_r2.append(r2_value)

            results_table[model + 25, split] = np.mean(list_r2)

    return results_table



print('Calculating...')

table = calculator_rmse_mcsqbc_r2(mcs=preds_mcs, qbc=preds_qbc, targets_mcs=targets_mcs)


print('Saving...')

pd.DataFrame(table[1:, :], columns=[table[0]]).to_csv('results.csv')
