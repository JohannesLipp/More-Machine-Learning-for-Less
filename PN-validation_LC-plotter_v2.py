import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.lines import Line2D

df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/7/results.csv', header=None)

data = np.asarray(df_data)[:, 1:]
trainingSetSizes = data[0, :-2]

def plot_learning_curve_rmse(data):

    train_sizes = trainingSetSizes
    plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples', labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    # -------->  MCS RR <------------------
    mcs_rr_mean = data[1, :-2]
    plt.plot(train_sizes, mcs_rr_mean, '.-', linestyle='--', color='maroon')
    # -------->  QBC RR <------------------
    qbc_rr_mean = data[7, :-2]
    plt.plot(train_sizes, qbc_rr_mean, '.-', linestyle='-.', color='firebrick')

    # -------->  MCS MLP <------------------
    mcs_mlp_mean = data[2, :-2]
    plt.plot(train_sizes, mcs_mlp_mean, '.-', linestyle='--', color='lightseagreen')
    # -------->  QBC MLP <------------------
    qbc_mlp_mean = data[8, :-2]
    plt.plot(train_sizes, qbc_mlp_mean, '.-', linestyle='-.', color='turquoise')

    # -------->  MCS XGB <------------------
    mcs_xbg_mean = data[3, :-2]
    plt.plot(train_sizes, mcs_xbg_mean, '.-', linestyle='--', color='peru')
    # -------->  QBC XGB <------------------
    qbc_xbg_mean = data[9, :-2]
    plt.plot(train_sizes, qbc_xbg_mean, '.-', linestyle='-.', color='burlywood')


    marks = [Line2D([0], [0], color='maroon', lw=3, label='Ridge Regression, MCS'),
             Line2D([0], [0], color='lightseagreen', lw=3, label='Multi-layer Perceptron, MCS'),
             Line2D([0], [0], color='peru', lw=3, label='XGBoost, MCS'),
             Line2D([0], [0], color='firebrick', lw=3, label='Ridge Regression, QBC'),
             Line2D([0], [0], color='turquoise', lw=3, label='Multi-layer Perceptron, QBC'),
             Line2D([0], [0], color='burlywood', lw=3, label='XGBoost, QBC')
             # Line2D([0], [0], color='black', linestyle='--', lw=1, label='Monte Carlo Sampling'),
             # Line2D([0], [0], color='black', linestyle='-.', lw=1, label='Query By Committee')
             # Line2D([0], [0], color='black', lw=1, label='Latin Hypercube Sampling'),
             # Line2D([0], [0], color='black', linestyle='--', lw=0, marker='*', label='Central Composite Design')
             ]


    plt.legend(handles=marks, loc='best')
    plt.ylabel('RMSE', labelpad=10)
    #plt.xscale('log')
    plt.xticks(train_sizes, train_sizes)
    return plt

def plot_learning_curve_std(data):

    train_sizes = trainingSetSizes
    plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples', labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    # -------->  MCS RR <------------------
    mcs_rr_mean = data[10, :-2]
    plt.plot(train_sizes, mcs_rr_mean, '.-', linestyle='--', color='maroon')
    # -------->  QBC RR <------------------
    qbc_rr_mean = data[16, :-2]
    plt.plot(train_sizes, qbc_rr_mean, '.-', linestyle='-.', color='firebrick')

    # -------->  MCS MLP <------------------
    mcs_mlp_mean = data[11, :-2]
    plt.plot(train_sizes, mcs_mlp_mean, '.-', linestyle='--', color='lightseagreen')
    # -------->  QBC MLP <------------------
    qbc_mlp_mean = data[17, :-2]
    plt.plot(train_sizes, qbc_mlp_mean, '.-', linestyle='-.', color='turquoise')

    # -------->  MCS XGB <------------------
    mcs_xbg_mean = data[12, :-2]
    plt.plot(train_sizes, mcs_xbg_mean, '.-', linestyle='--', color='peru')
    # -------->  QBC XGB <------------------
    qbc_xbg_mean = data[18, :-2]
    plt.plot(train_sizes, qbc_xbg_mean, '.-', linestyle='-.', color='burlywood')

    marks = [Line2D([0], [0], color='maroon', lw=3, label='Ridge Regression, MCS'),
             Line2D([0], [0], color='lightseagreen', lw=3, label='Multi-layer Perceptron, MCS'),
             Line2D([0], [0], color='peru', lw=3, label='XGBoost, MCS'),
             Line2D([0], [0], color='firebrick', lw=3, label='Ridge Regression, QBC'),
             Line2D([0], [0], color='turquoise', lw=3, label='Multi-layer Perceptron, QBC'),
             Line2D([0], [0], color='burlywood', lw=3, label='XGBoost, QBC')
             # Line2D([0], [0], color='black', linestyle='--', lw=1, label='Monte Carlo Sampling'),
             # Line2D([0], [0], color='black', linestyle='-.', lw=1, label='Query By Committee')
             # Line2D([0], [0], color='black', lw=1, label='Latin Hypercube Sampling'),
             # Line2D([0], [0], color='black', linestyle='--', lw=0, marker='*', label='Central Composite Design')
             ]

    plt.legend(handles=marks, loc='best')
    plt.ylabel('STD of RMSEs', labelpad=10)
    #plt.xscale('log')
    plt.xticks(train_sizes, train_sizes)
    return plt

def plot_learning_curve_r2(data):

    train_sizes = trainingSetSizes
    plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples', labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    # -------->  MCS RR <------------------
    mcs_rr_mean = data[19, :-2]
    plt.plot(train_sizes, mcs_rr_mean, '.-', linestyle='--', color='maroon')
    # -------->  QBC RR <------------------
    qbc_rr_mean = data[25, :-2]
    plt.plot(train_sizes, qbc_rr_mean, '.-', linestyle='-.', color='firebrick')

    # -------->  MCS MLP <------------------
    mcs_mlp_mean = data[20, :-2]
    plt.plot(train_sizes, mcs_mlp_mean, '.-', linestyle='--', color='lightseagreen')
    # -------->  QBC MLP <------------------
    qbc_mlp_mean = data[26, :-2]
    plt.plot(train_sizes, qbc_mlp_mean, '.-', linestyle='-.', color='turquoise')

    # -------->  MCS XGB <------------------
    mcs_xbg_mean = data[21, :-2]
    plt.plot(train_sizes, mcs_xbg_mean, '.-', linestyle='--', color='peru')
    # -------->  QBC XGB <------------------
    qbc_xbg_mean = data[27, :-2]
    plt.plot(train_sizes, qbc_xbg_mean, '.-', linestyle='-.', color='burlywood')

    marks = [Line2D([0], [0], color='maroon', lw=3, label='Ridge Regression, MCS'),
             Line2D([0], [0], color='lightseagreen', lw=3, label='Multi-layer Perceptron, MCS'),
             Line2D([0], [0], color='peru', lw=3, label='XGBoost, MCS'),
             Line2D([0], [0], color='firebrick', lw=3, label='Ridge Regression, QBC'),
             Line2D([0], [0], color='turquoise', lw=3, label='Multi-layer Perceptron, QBC'),
             Line2D([0], [0], color='burlywood', lw=3, label='XGBoost, QBC')
             # Line2D([0], [0], color='black', linestyle='--', lw=1, label='Monte Carlo Sampling'),
             # Line2D([0], [0], color='black', linestyle='-.', lw=1, label='Query By Committee')
             # Line2D([0], [0], color='black', lw=1, label='Latin Hypercube Sampling'),
             # Line2D([0], [0], color='black', linestyle='--', lw=0, marker='*', label='Central Composite Design')
             ]

    plt.legend(handles=marks, loc='best')
    plt.ylabel('R2 score [-]', labelpad=10)
    #plt.xscale('log')
    plt.xticks(train_sizes, train_sizes)
    return plt


# -------->  PLOTS <------------------
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '13'})
rc('text', usetex=True)

fig1 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve_rmse(data)
plt.title('RMSE, average of 20 test runs (20 different testsets)')
plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples, bootstrap=True, committee.rebag()', labelpad=10)
#plt.xscale('log')
#plt.xlim([5, 360])
plt.xticks([10, 50, 100, 150, 200, 250, 300, 350], [10, 50, 100, 150, 200, 250, 300, 350])
plt.ylim([-0.2, 5.2])
plt.grid()
fig1.savefig('AUTO_MPG_RMSE_20test_bootstrap,rebag_alnorm.png', dpi=300)
'''
fig2 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve_std(data)
plt.title('Standard deviation of RMSEs')
plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples, bootstrap=True', labelpad=10)
#plt.xscale('log')
#plt.xlim([5, 360])
plt.xticks([9, 54, 99, 153, 198, 252, 297, 351], [9, 54, 99, 153, 198, 252, 297, 351])
#plt.ylim([0, 0.085])
plt.grid()
#fig2.savefig('AUTO_MPG_STD_5test5train.png', dpi=300)

fig3 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve_r2(data)
plt.title('R2 score')
plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples, bootstrap=True', labelpad=10)
#plt.xscale('log')
#plt.xlim([5, 360])
plt.xticks([9, 54, 99, 153, 198, 252, 297, 351], [9, 54, 99, 153, 198, 252, 297, 351])
plt.ylim([-0.25, 1])
plt.grid()
#fig3.savefig('AUTO_MPG_R2_5test5train.png', dpi=300)
'''
plt.show()