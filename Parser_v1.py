import numpy as np
import pandas as pd
import glob
from sklearn.metrics import r2_score
from matplotlib import rc
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------

batchSize = 20  # size of the training set increments
seedSize = 10  # size of the initial training set
num_iters_test = 10  # choose desired number of cv iterations
size_test_set = 0.1  # choose desired size of the test set
numCommitteeMembers = 5  # choose number of committee members for the QBC
predictors = ['Ridge', 'MLPRegressor', 'XGBRegressor']


# -------->  FUNCTIONS  <------------------
# RMSE CALCULATION
def rmse(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2.
    return np.sqrt(residuals / n)


# parsing the txt files into a DataFrame
def results_parser(path):
    globsearch = path + '/**/part-*'
    fileNameList = []

    #   for filename in glob.iglob('/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/VS_Results/**/part-*', recursive=True):
    for filename in glob.iglob(globsearch, recursive=True):
        fileNameList.append(filename)

    assert (len(fileNameList) > 0)

    columnNames = []
    for predictor in predictors:
        columnNames.append(predictor + '_targets')
        columnNames.append(predictor + '_preds')

    df_results = pd.DataFrame(columns=columnNames, index=np.arange(num_iters_test), dtype=object)

    for predictor in predictors:
        for fold in range(num_iters_test):
            for filename in fileNameList:
                foldstr = 'fold' + str(fold + 1) + '/'
                if predictor in filename and foldstr in filename:
                    if 'targets' in filename:
                        print(filename)
                        columnName = predictor + '_targets'
                        df_results.loc[fold, columnName] = np.loadtxt(filename).reshape(-1, 1)
                    else:
                        print(filename)
                        columnName = predictor + '_preds'
                        df_results.loc[fold, columnName] = np.loadtxt(filename).reshape(-1, 1)

    return df_results


# -------->  IMPORT UTILIZED DATASET FOR FURTHER INFORMATION <------------------
df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/MLPlatform/data/auto_mpg.csv', header=0)
df_data = df_data.fillna(value=0)
arr_data = np.asarray(df_data)

len_training_set = int(arr_data.shape[0] * (1 - size_test_set))
len_test_set = arr_data.shape[0] - len_training_set


def calculate_RMSE(dataframe):
    columnNames = [seedSize] + list(np.arange(batchSize + seedSize, len_training_set, batchSize))
    assert (len(dataframe.iloc[0, 0]) == len(columnNames) * len_test_set)

    df_RMSEs = pd.DataFrame(columns=columnNames, index=predictors, data=np.nan)

    for predictor in predictors:
        for idx, split in enumerate(columnNames):
            listRMSEs = []
            for fold in range(num_iters_test):
                targetName = predictor + '_targets'
                predsName = predictor + '_preds'
                startIdx = idx * len_test_set
                endIdx = idx * len_test_set + len_test_set
                targets = dataframe.loc[fold, targetName][startIdx: endIdx]
                predictions = dataframe.loc[fold, predsName][startIdx: endIdx]
                valueRMSE = rmse(targets, predictions)
                listRMSEs.append(valueRMSE)

            df_RMSEs.loc[predictor, split] = np.mean(listRMSEs)

    return df_RMSEs


def calculate_STD(dataframe):
    columnNames = [seedSize] + list(np.arange(batchSize + seedSize, len_training_set, batchSize))
    assert (len(dataframe.iloc[0, 0]) == len(columnNames) * len_test_set)

    df_STDs = pd.DataFrame(columns=columnNames, index=predictors, data=np.nan)

    for predictor in predictors:
        for idx, split in enumerate(columnNames):
            listRMSEs = []
            for fold in range(num_iters_test):
                targetName = predictor + '_targets'
                predsName = predictor + '_preds'
                startIdx = idx * len_test_set
                endIdx = idx * len_test_set + len_test_set
                targets = dataframe.loc[fold, targetName][startIdx: endIdx]
                predictions = dataframe.loc[fold, predsName][startIdx: endIdx]
                valueRMSE = rmse(targets, predictions)
                listRMSEs.append(valueRMSE)

            df_STDs.loc[predictor, split] = np.std(listRMSEs)

    return df_STDs


def calculate_R2score(dataframe):
    columnNames = [seedSize] + list(np.arange(batchSize + seedSize, len_training_set, batchSize))
    assert (len(dataframe.iloc[0, 0]) == len(columnNames) * len_test_set)

    df_R2s = pd.DataFrame(columns=columnNames, index=predictors, data=np.nan)

    for predictor in predictors:
        for idx, split in enumerate(columnNames):
            listR2s = []
            for fold in range(num_iters_test):
                targetName = predictor + '_targets'
                predsName = predictor + '_preds'
                startIdx = idx * len_test_set
                endIdx = idx * len_test_set + len_test_set
                targets = dataframe.loc[fold, targetName][startIdx: endIdx]
                predictions = dataframe.loc[fold, predsName][startIdx: endIdx]
                valueR2 = r2_score(targets, predictions)
                listR2s.append(valueR2)

            df_R2s.loc[predictor, split] = np.mean(listR2s)

    return df_R2s


def plot_learning_curve(data):

    models = {'Ridge':'maroon', 'MLPRegressor':'lightseagreen', 'XGBRegressor':'peru'}
    train_sizes = data.columns.values
    plt.xlabel('Number of samples inluded in training set\ndataset size: {} samples'.format(arr_data.shape[0]), labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    for predictor in predictors:
        values = data.loc[predictor, :].tolist()
        print(values)
        print(type(values))
        plt.plot(train_sizes, values, '.-', linestyle='-.', color=models[predictor])


    marks = [Line2D([0], [0], color='maroon', lw=3, label='Ridge Regression'),
             Line2D([0], [0], color='lightseagreen', lw=3, label='Multi-layer Perceptron'),
             Line2D([0], [0], color='peru', lw=3, label='XGBoost'),
             # Line2D([0], [0], color='black', linestyle='--', lw=1, label='Monte Carlo Sampling'),
             Line2D([0], [0], color='black', linestyle='-.', lw=1, label='Query By Committee')
             # Line2D([0], [0], color='black', lw=1, label='Latin Hypercube Sampling'),
             # Line2D([0], [0], color='black', linestyle='--', lw=0, marker='*', label='Central Composite Design')
             ]

    plt.legend(handles=marks, loc='best')
    plt.ylabel('RMSE [g]', labelpad=10)
    # plt.xscale('log')
    plt.xticks(train_sizes, train_sizes)
    return plt



df = results_parser('/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/VS_Results')

df_RMSE = calculate_RMSE(df)
df_STD = calculate_STD(df)
df_R2 = calculate_R2score(df)


# -------->  PLOTS <------------------
# plt.style.use('seaborn-dark')  # plt.style.use('seaborn-white')
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '13'})
# rc('text', usetex=True)

fig1 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(df_RMSE)
plt.title('RMSE')
plt.ylim([0, 6.2])
plt.grid()
fig1.savefig('VS_spark_RMSE_v1.png', dpi=300)

fig2 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(df_STD)
plt.title('STD')
plt.ylim([0, 1.4])
plt.grid()
fig2.savefig('VS_spark_STD_v1.png', dpi=300)

fig3 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(df_R2)
plt.title('R2 score')
plt.ylim([0, 1.05])
plt.grid()
fig3.savefig('VS_spark_R2score_v1.png', dpi=300)

plt.show()


print('done')
