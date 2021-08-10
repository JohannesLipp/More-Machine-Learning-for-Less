import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.lines import Line2D



df_data = pd.read_csv(
    '/Users/philippnoodt/Jobs_Bewerbungen/IMA/Python/AL_Kunststoff/2019-03-14/3/results.csv', header=None)

data = np.asarray(df_data)[:, 1:]
splits = data[0, :-2]
sams = np.round(splits * 392 * 0.9)
sams = np.round(splits * 392)

def plot_learning_curve_rmse(data):

    train_sizes = sams
    # [4, 6, 9, 14, 20, 31, 46, 69, 103, 155, 233, 350, 525]
    plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples', labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    # -------->  MCS RR <------------------
    mcs_rr_mean = data[1, :-2]
    plt.plot(train_sizes, mcs_rr_mean, '.-', linestyle='--', color='maroon')
    # -------->  LHS RR <------------------
    # lhs_rr_mean = data[4, :-2]
    # plt.plot(train_sizes, lhs_rr_mean, '.-', color='maroon')
    # -------->  QBC RR <------------------
    qbc_rr_mean = data[7, :-2]
    plt.plot(train_sizes, qbc_rr_mean, '.-', linestyle='-.', color='firebrick')
    # -------->  CCD RR <------------------
    # plt.scatter(data[0, -1], data[5, -1], color='maroon', marker='*', s=50)

    # -------->  MCS MLP <------------------
    mcs_mlp_mean = data[2, :-2]
    plt.plot(train_sizes, mcs_mlp_mean, '.-', linestyle='--', color='lightseagreen')
    # -------->  LHS MLP <------------------
    # lhs_mlp_mean = data[5, :-2]
    # plt.plot(train_sizes, lhs_mlp_mean, '.-', color='lightseagreen')
    # -------->  QBC MLP <------------------
    qbc_mlp_mean = data[8, :-2]
    plt.plot(train_sizes, qbc_mlp_mean, '.-', linestyle='-.', color='turquoise')
    # -------->  CCD MLP <------------------
    # plt.scatter(data[0, -1], data[1, -1], color='lightseagreen', marker='*', s=50)

    # -------->  MCS XGB <------------------
    mcs_xbg_mean = data[3, :-2]
    plt.plot(train_sizes, mcs_xbg_mean, '.-', linestyle='--', color='peru')
    # -------->  LHS XGB <------------------
    # lhs_xgb_mean = data[6, :-2]
    # plt.plot(train_sizes, lhs_xgb_mean, '.-', color='peru')
    # -------->  QBC XGB <------------------
    qbc_xbg_mean = data[9, :-2]
    plt.plot(train_sizes, qbc_xbg_mean, '.-', linestyle='-.', color='burlywood')
    # -------->  CCD XGB <------------------
    #plt.scatter(data[0, -1], data[3, -1], color='peru', marker='*', s=50)

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

    '''
    if doc != 1:
        # -------->  BBD RR <------------------
        plt.scatter(data[0, -2], data[5, -2], color='maroon', marker='v', s=50)
        # -------->  BBD MLP <------------------
        plt.scatter(data[0, -2], data[1, -2], color='lightseagreen', marker='v', s=50)
        # -------->  BBD XGB <------------------
        plt.scatter(data[0, -2], data[3, -2], color='peru', marker='v', s=50)

        marks = [Line2D([0], [0], color='maroon', lw=3, label='Ridge Regression'),
                 Line2D([0], [0], color='lightseagreen', lw=3, label='Multi-layer Perceptron'),
                 Line2D([0], [0], color='peru', lw=3, label='XGBoost'),
                 Line2D([0], [0], color='black', linestyle='--', lw=1, label='Monte Carlo Sampling'),
                 Line2D([0], [0], color='black', lw=1, label='Latin Hypercube Sampling'),
                 #Line2D([0], [0], color='black', linestyle='--', lw=0, marker='v', label='Box-Behnken Design'),
                 Line2D([0], [0], color='black', linestyle='-.', lw=1, label='Query By Committee')]
    '''
    plt.legend(handles=marks, loc='best')
    plt.ylabel('RMSE', labelpad=10)
    #plt.xscale('log')
    plt.xticks(train_sizes, train_sizes)
    return plt

def plot_learning_curve_std(data):

    train_sizes = sams
    # [4, 6, 9, 14, 20, 31, 46, 69, 103, 155, 233, 350, 525]
    plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples', labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    # -------->  MCS RR <------------------
    mcs_rr_mean = data[10, :-2]
    plt.plot(train_sizes, mcs_rr_mean, '.-', linestyle='--', color='maroon')
    # -------->  LHS RR <------------------
    # lhs_rr_mean = data[4, :-2]
    # plt.plot(train_sizes, lhs_rr_mean, '.-', color='maroon')
    # -------->  QBC RR <------------------
    qbc_rr_mean = data[16, :-2]
    plt.plot(train_sizes, qbc_rr_mean, '.-', linestyle='-.', color='maroon')
    # -------->  CCD RR <------------------
    # plt.scatter(data[0, -1], data[5, -1], color='maroon', marker='*', s=50)

    # -------->  MCS MLP <------------------
    mcs_mlp_mean = data[11, :-2]
    plt.plot(train_sizes, mcs_mlp_mean, '.-', linestyle='--', color='lightseagreen')
    # -------->  LHS MLP <------------------
    # lhs_mlp_mean = data[5, :-2]
    # plt.plot(train_sizes, lhs_mlp_mean, '.-', color='lightseagreen')
    # -------->  QBC MLP <------------------
    qbc_mlp_mean = data[17, :-2]
    plt.plot(train_sizes, qbc_mlp_mean, '.-', linestyle='-.', color='lightseagreen')
    # -------->  CCD MLP <------------------
    # plt.scatter(data[0, -1], data[1, -1], color='lightseagreen', marker='*', s=50)

    # -------->  MCS XGB <------------------
    mcs_xbg_mean = data[12, :-2]
    plt.plot(train_sizes, mcs_xbg_mean, '.-', linestyle='--', color='peru')
    # -------->  LHS XGB <------------------
    # lhs_xgb_mean = data[6, :-2]
    # plt.plot(train_sizes, lhs_xgb_mean, '.-', color='peru')
    # -------->  QBC XGB <------------------
    qbc_xbg_mean = data[18, :-2]
    plt.plot(train_sizes, qbc_xbg_mean, '.-', linestyle='-.', color='peru')
    # -------->  CCD XGB <------------------
    #plt.scatter(data[0, -1], data[3, -1], color='peru', marker='*', s=50)

    marks = [Line2D([0], [0], color='maroon', lw=3, label='Ridge Regression'),
             Line2D([0], [0], color='lightseagreen', lw=3, label='Multi-layer Perceptron'),
             Line2D([0], [0], color='peru', lw=3, label='XGBoost'),
             Line2D([0], [0], color='black', linestyle='--', lw=1, label='Monte Carlo Sampling'),
             Line2D([0], [0], color='black', linestyle='-.', lw=1, label='Query By Committee')
             # Line2D([0], [0], color='black', lw=1, label='Latin Hypercube Sampling'),
             # Line2D([0], [0], color='black', linestyle='--', lw=0, marker='*', label='Central Composite Design')
             ]

    plt.legend(handles=marks, loc='best')
    plt.ylabel('STD of RMSEs', labelpad=10)
    #plt.xscale('log')
    plt.xticks(train_sizes, train_sizes)
    return plt

def plot_learning_curve_r2(data):

    train_sizes = sams
    # [4, 6, 9, 14, 20, 31, 46, 69, 103, 155, 233, 350, 525]
    plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples', labelpad=10)

    print('train sizes: ', train_sizes)
    print('done: learning_curve() calculations')

    # -------->  MCS RR <------------------
    mcs_rr_mean = data[19, :-2]
    plt.plot(train_sizes, mcs_rr_mean, '.-', linestyle='--', color='maroon')
    # -------->  LHS RR <------------------
    # lhs_rr_mean = data[4, :-2]
    # plt.plot(train_sizes, lhs_rr_mean, '.-', color='maroon')
    # -------->  QBC RR <------------------
    qbc_rr_mean = data[25, :-2]
    plt.plot(train_sizes, qbc_rr_mean, '.-', linestyle='-.', color='maroon')
    # -------->  CCD RR <------------------
    # plt.scatter(data[0, -1], data[5, -1], color='maroon', marker='*', s=50)

    # -------->  MCS MLP <------------------
    mcs_mlp_mean = data[20, :-2]
    plt.plot(train_sizes, mcs_mlp_mean, '.-', linestyle='--', color='lightseagreen')
    # -------->  LHS MLP <------------------
    # lhs_mlp_mean = data[5, :-2]
    # plt.plot(train_sizes, lhs_mlp_mean, '.-', color='lightseagreen')
    # -------->  QBC MLP <------------------
    qbc_mlp_mean = data[26, :-2]
    plt.plot(train_sizes, qbc_mlp_mean, '.-', linestyle='-.', color='lightseagreen')
    # -------->  CCD MLP <------------------
    # plt.scatter(data[0, -1], data[1, -1], color='lightseagreen', marker='*', s=50)

    # -------->  MCS XGB <------------------
    mcs_xbg_mean = data[21, :-2]
    plt.plot(train_sizes, mcs_xbg_mean, '.-', linestyle='--', color='peru')
    # -------->  LHS XGB <------------------
    # lhs_xgb_mean = data[6, :-2]
    # plt.plot(train_sizes, lhs_xgb_mean, '.-', color='peru')
    # -------->  QBC XGB <------------------
    qbc_xbg_mean = data[27, :-2]
    plt.plot(train_sizes, qbc_xbg_mean, '.-', linestyle='-.', color='peru')
    # -------->  CCD XGB <------------------
    #plt.scatter(data[0, -1], data[3, -1], color='peru', marker='*', s=50)

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
# plt.style.use('seaborn-dark')  # plt.style.use('seaborn-white')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '13'})
rc('text', usetex=True)

fig1 = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.16)
plot_learning_curve_rmse(data)
plt.title('RMSE, average of 20 test runs ')
plt.xlabel('Number of samples inluded in training set\ndataset size: 392 samples, bootstrap=True', labelpad=10)
#plt.xscale('log')
#plt.xlim([5, 360])
plt.xticks([10, 50, 100, 150, 200, 250, 300, 350, 390], [10, 50, 100, 150, 200, 250, 300, 350, 390])
plt.ylim([-0.2, 5.2])
plt.grid()
# fig1.savefig('AUTO_MPG_RMSE_20test_bootstrap_alnorm.png', dpi=300)
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