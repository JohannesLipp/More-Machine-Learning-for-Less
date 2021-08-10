import numpy as np
import pandas as pd
import glob
from sklearn.metrics import r2_score
from matplotlib import rc
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import datetime

# -------->  INSTRUCTIONS  <------------------
# 1. enter correct correct path to results directory
# 2. run discover_val_parameters(path) method and store the generated lists of used predictors, samplers, folds and
#   training sizes
# 3. run RMSE_table_calculator(), R2score_table_calculator() and STD_table_calculator() with inputs from (2.) in order
#   to generate three separate tables containing the datapoints for the plots
# 4. use plot_learning_curve(data, samplers, predictor, criterium) to plot either of the three criteria RMSE, R2score
#   or STD. Enter list of samplers (from 2.), and both the predictor and criterium to be plotted

# -------->  ASSIGNMENT INPUT VARIABLES  <------------------
path = 'C:/SVN/Papers/DOE_for_ML_Paper/03_Results/Results_AL_train2000_test_3000_increment_1/'

# -------->  DISCOVERY OF VALIDATION PARAMETERS  <------------------

def discover_val_parameters(path):
    predictors = []
    samplers = []
    folds = []
    training_sizes = []

    for filename in glob.iglob(path + '/**/', recursive=True):
        #fileInfo = filename.split(path)[-1].split('_')
        fileInfo = filename[len(path):].split('_')
        try:
            if fileInfo[1] not in predictors:
                predictors.append(fileInfo[1])
            if fileInfo[2] not in samplers:
                samplers.append(fileInfo[2])
            if int(fileInfo[4]) not in folds:
                folds.append(int(fileInfo[4]))
            if int(fileInfo[-1][:-1]) not in training_sizes:
                training_sizes.append(int(fileInfo[-1][:-1]))
        except:
            pass

    predictors.sort()
    samplers.sort()
    folds.sort()
    training_sizes.sort()

    return predictors, samplers, folds, training_sizes


# -------->  FUNCTIONS  <------------------
# RMSE CALCULATION
def rmse(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2.
    return np.sqrt(residuals / n)


# CREATION OF THE RMSE RESULTS TABLE
def RMSE_table_calculator(predictors, samplers, folds, training_sizes):
    df_RMSE = pd.DataFrame(columns=training_sizes, index=np.arange(len(predictors)*len(samplers)), data=np.nan)
    df_idx = 0
    failed_Imports = []
    for sampler in samplers:
        for predictor in predictors:
            for training_size in training_sizes:
                fold_RMSEs = []
                for fold in folds:

                    # -------->  Import of prediction and target files  <------------------
                    predsName = 'preds_' + predictor + '_' + sampler + '_fold_' + str(fold) + '_taining_size_' + str(
                        training_size) + '/part-*'
                    # print(predsName)
                    preds = np.array([])
                    predsList = glob.glob(path + predsName)
                    if len(predsList) == 1:
                        # print(' -> predsFile exists.')
                        full_path = predsList[0]
                        try:
                            preds = np.loadtxt(full_path).reshape(-1, 1)
                        except:
                            print('Could not load: ' + predsName)
                            failed_Imports.append(predsName)
                    else:
                        print(' There is(are) ' + str(len(predsList)) + ' file(s) called ' + predsName)

                    targetsName = 'targets_' + predictor + '_' + sampler + '_fold_' + str(fold) + '_taining_size_' + str(
                        training_size) + '/part-*'
                    # print(targetsName)
                    targets = np.array([])
                    targetsList = glob.glob(path + targetsName)
                    if len(targetsList) == 1:
                        # print(' -> targetsFile exists.')
                        full_path = targetsList[0]
                        try:
                            targets = np.loadtxt(full_path).reshape(-1, 1)
                        except:
                            print('Could not load: ' + targetsName)
                            failed_Imports.append(targetsName)
                    else:
                        print(' There is(are) ' + str(len(targetsList)) + ' file(s) called ' + targetsName)


                    # --------> fold RMSE calculation  <------------------
                    if preds.size != 0 and targets.size != 0:
                        assert (preds.shape == targets.shape)
                        fold_RMSEs.append(rmse(targets, preds))

                    else:
                        print('Either the preds array or the targets array was not imported.')

                # --------> mean fold RMSE calculation  <------------------
                df_RMSE.loc[df_idx, training_size] = np.mean(fold_RMSEs)

            df_idx += 1
            print(df_idx)

    print(df_RMSE)

    return(df_RMSE)



# CREATION OF THE R2 SCORE RESULTS TABLE
def R2score_table_calculator(predictors, samplers, folds, training_sizes):
    df_R2score = pd.DataFrame(columns=training_sizes, index=np.arange(len(predictors) * len(samplers)), data=np.nan)
    df_idx = 0
    failed_Imports = []
    for sampler in samplers:
        for predictor in predictors:
            for training_size in training_sizes:
                fold_R2scores = []
                for fold in folds:

                    # -------->  Import of prediction and target files  <------------------
                    predsName = 'preds_' + predictor + '_' + sampler + '_fold_' + str(fold) + '_taining_size_' + str(
                        training_size) + '/part-*'
                    # print(predsName)
                    preds = np.array([])
                    predsList = glob.glob(path + predsName)
                    if len(predsList) == 1:
                        # print(' -> predsFile exists.')
                        full_path = predsList[0]
                        try:
                            preds = np.loadtxt(full_path).reshape(-1, 1)
                        except:
                            print('Could not load: ' + predsName)
                            failed_Imports.append(predsName)
                    else:
                        print(' There is(are) ' + str(len(predsList)) + ' file(s) called ' + predsName)

                    targetsName = 'targets_' + predictor + '_' + sampler + '_fold_' + str(
                        fold) + '_taining_size_' + str(
                        training_size) + '/part-*'
                    # print(targetsName)
                    targets = np.array([])
                    targetsList = glob.glob(path + targetsName)
                    if len(targetsList) == 1:
                        # print(' -> targetsFile exists.')
                        full_path = targetsList[0]
                        try:
                            targets = np.loadtxt(full_path).reshape(-1, 1)
                        except:
                            print('Could not load: ' + targetsName)
                            failed_Imports.append(targetsName)
                    else:
                        print(' There is(are) ' + str(len(targetsList)) + ' file(s) called ' + targetsName)

                    # --------> fold R2score calculation  <------------------
                    if preds.size != 0 and targets.size != 0:
                        assert (preds.shape == targets.shape)
                        fold_R2scores.append(r2_score(targets, preds))

                    else:
                        print('Either the preds array or the targets array was not imported.')

                # --------> mean fold R2score calculation  <------------------
                df_R2score.loc[df_idx, training_size] = np.mean(fold_R2scores)

            df_idx += 1
            print(df_idx)

    print(df_R2score)

    return (df_R2score)



# CREATION OF THE STD RESULTS TABLE
def STD_table_calculator(predictors, samplers, folds, training_sizes):
    df_std = pd.DataFrame(columns=training_sizes, index=np.arange(len(predictors) * len(samplers)), data=np.nan)
    df_idx = 0
    failed_Imports = []
    for sampler in samplers:
        for predictor in predictors:
            for training_size in training_sizes:
                fold_rmses = []
                for fold in folds:

                    # -------->  Import of prediction and target files  <------------------
                    predsName = 'preds_' + predictor + '_' + sampler + '_fold_' + str(fold) + '_taining_size_' + str(
                        training_size) + '/part-*'
                    # print(predsName)
                    preds = np.array([])
                    predsList = glob.glob(path + predsName)
                    if len(predsList) == 1:
                        # print(' -> predsFile exists.')
                        full_path = predsList[0]
                        try:
                            preds = np.loadtxt(full_path).reshape(-1, 1)
                        except:
                            print('Could not load: ' + predsName)
                            failed_Imports.append(predsName)
                    else:
                        print(' There is(are) ' + str(len(predsList)) + ' file(s) called ' + predsName)

                    targetsName = 'targets_' + predictor + '_' + sampler + '_fold_' + str(
                        fold) + '_taining_size_' + str(
                        training_size) + '/part-*'
                    # print(targetsName)
                    targets = np.array([])
                    targetsList = glob.glob(path + targetsName)
                    if len(targetsList) == 1:
                        # print(' -> targetsFile exists.')
                        full_path = targetsList[0]
                        try:
                            targets = np.loadtxt(full_path).reshape(-1, 1)
                        except:
                            print('Could not load: ' + targetsName)
                            failed_Imports.append(targetsName)
                    else:
                        print(' There is(are) ' + str(len(targetsList)) + ' file(s) called ' + targetsName)

                    # --------> fold std calculation  <------------------
                    if preds.size != 0 and targets.size != 0:
                        assert (preds.shape == targets.shape)
                        fold_rmses.append(rmse(targets, preds))

                    else:
                        print('Either the preds array or the targets array was not imported.')

                # --------> mean fold std calculation  <------------------
                df_std.loc[df_idx, training_size] = np.std(fold_rmses)

            df_idx += 1
            print(df_idx)

    print(df_std)

    return (df_std)



def plot_learning_curve(data, samplers, predictor, criterium):
    if predictor == 'MLPRegressor':
        df_idxs = [0, 3, 6]
    elif predictor == 'Ridge':
        df_idxs = [1, 4, 7]
    else:
        df_idxs = [2, 5, 8]

    sampler_dict = {}
    colors = ['lightseagreen', 'maroon', 'peru']
    marks = []

    train_sizes = [int(val) for val in rmse_table.columns.values]


    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')
    print(predictor)
    for idx, sampler in enumerate(samplers):
        sampler_dict[sampler] = colors[idx]
        values = data.iloc[df_idxs[idx], :].tolist()
        print(values, sampler_dict[sampler])
        plt.plot(train_sizes, values, '.-', linestyle='-.', color=sampler_dict[sampler])

        marks.append(Line2D([0], [0], color=sampler_dict[sampler], lw=3, label=sampler))


    plt.legend(handles=marks, loc='best')
    criteria_dict = {'RMSE': '[unit]',
                     'STD': '[unit]',
                     'R2score': '[-]'}

    # plt.title(criterium + ', ' + predictor)
    # plt.xlabel('Number of samples inluded in training set', labelpad=10)
    # plt.ylabel(criterium + ' ' + criteria_dict[criterium], labelpad=10)

    if criterium == 'RMSE':
        plt.ylim([0, 7])
    elif criterium == 'R2score':
        plt.ylim([0.8, 1.01])
    elif criterium == 'STD':
        plt.ylim([0, 1.4])
    # plt.xscale('log')
    plt.xticks(np.arange(0, 1000, 250))
    plt.grid()

    return plt



val_paras = discover_val_parameters(path)

rmse_table = RMSE_table_calculator(val_paras[0], val_paras[1], val_paras[2], val_paras[3])
# rmse_table.to_csv('RMSE_results_v1')

r2score_table = R2score_table_calculator(val_paras[0], val_paras[1], val_paras[2], val_paras[3])
# r2score_table.to_csv('R2score_results_v1')

std_table = STD_table_calculator(val_paras[0], val_paras[1], val_paras[2], val_paras[3])
# std_table.to_csv('std_results_v1')


'''
rmse_table = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/VS_Results/2019-05-16/rmse_results_v1',
    header=0, index_col=0)
r2score_table = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/VS_Results/2019-05-16/r2score_results_v1',
    header=0, index_col=0)
std_table = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/VS_Results/2019-05-16/std_results_v1',
    header=0, index_col=0)
'''

# -------->  PLOTS <------------------
# plt.style.use('seaborn-dark')  # plt.style.use('seaborn-white')
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '13'})
# rc('text', usetex=True)
'''
fig11 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(rmse_table, val_paras[1], 'MLPRegressor', 'RMSE')
fig11.savefig('MLP_RMSE_v1.png', dpi=300)

fig12 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(r2score_table, val_paras[1], 'MLPRegressor', 'R2score')
fig12.savefig('MLP_R2score_v1.png', dpi=300)

fig13 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(std_table, val_paras[1], 'MLPRegressor', 'STD')
fig13.savefig('MLP_STD_v1.png', dpi=300)

fig21 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(rmse_table, val_paras[1], 'XGBRegressor', 'RMSE')
fig21.savefig('XGB_RMSE_v1.png', dpi=300)

fig22 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(r2score_table, val_paras[1], 'XGBRegressor', 'R2score')
fig22.savefig('XGB_R2score_v1.png', dpi=300)

fig23 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(std_table, val_paras[1], 'XGBRegressor', 'STD')
fig23.savefig('XGB_STD_v1.png', dpi=300)

fig31 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(rmse_table, val_paras[1], 'Ridge', 'RMSE')
fig31.savefig('Ridge_RMSE_v1.png', dpi=300)

fig32 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(r2score_table, val_paras[1], 'Ridge', 'R2score')
fig32.savefig('Ridge_R2score_v1.png', dpi=300)

fig33 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve(std_table, val_paras[1], 'Ridge', 'STD')
fig33.savefig('Ridge_STD_v1.png', dpi=300)



fig101 = plt.figure(figsize=(10, 8))
fig101.subplots_adjust(left=0.11, bottom=0.11, right=0.97, top=0.93 , wspace=0.19, hspace=0.26)

plt.subplot(3, 3, 1)
plot_learning_curve(rmse_table, val_paras[1], 'MLPRegressor', 'RMSE')
plt.subplot(3, 3, 2)
plot_learning_curve(rmse_table, val_paras[1], 'XGBRegressor', 'RMSE')
plt.subplot(3, 3, 3)
plot_learning_curve(rmse_table, val_paras[1], 'Ridge', 'RMSE')
plt.subplot(3, 3, 4)
plot_learning_curve(r2score_table, val_paras[1], 'MLPRegressor', 'R2score')
plt.subplot(3, 3, 5)
plot_learning_curve(r2score_table, val_paras[1], 'XGBRegressor', 'R2score')
plt.subplot(3, 3, 6)
plot_learning_curve(r2score_table, val_paras[1], 'Ridge', 'R2score')
plt.subplot(3, 3, 7)
plot_learning_curve(std_table, val_paras[1], 'MLPRegressor', 'STD')
plt.subplot(3, 3, 8)
plot_learning_curve(std_table, val_paras[1], 'XGBRegressor', 'STD')
plt.subplot(3, 3, 9)
plot_learning_curve(std_table, val_paras[1], 'Ridge', 'STD')

fig101.text(0.03, 0.79, 'RMSE [unit]', va='center', rotation='vertical')
fig101.text(0.03, 0.53, '$R^{2}$ score [-]', va='center', rotation='vertical')
fig101.text(0.03, 0.25, 'STD [unit]', va='center', rotation='vertical')
fig101.text(0.22, 0.95, 'MLPRegressor', ha='center')
fig101.text(0.53, 0.95, 'XGBRegressor', ha='center')
fig101.text(0.84, 0.95, 'Ridge', ha='center')
fig101.text(0.5, 0.04, 'X-axes: number of samples inluded in training set', ha='center')

fig101.savefig('RMSE_R2score_STD_v1.png', dpi=900)
'''

plt.show()
