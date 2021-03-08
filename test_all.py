# import keras
# import numpy as np
# np.set_printoptions(threshold=np.inf)
# import h5py
# from keras.models import load_model
# import time
# import config
# config = config.Config()
#
#
#
# test2_records = np.load('ECG_train_data_process_3600QRS.npy')
# test_label = np.load('1Record_Label.npy')
#
# print(len(test2_records))
# print(len(test2_records[0]))
#
# print("-------------------------")
# # print(test2_records[:][0])
# # print(test2_records[:][0].shape)
# data = []
# max_data = 17.881176
# min_data = -14.625054
# k = (config.normalization_max - config.normalization_min)/((max_data - min_data) * 1.0) # 比例系数
# tic = time.time()
# print("start time：",tic)
# for i in range(len(test2_records)):
#
#     test2_records[i][:,0] = config.normalization_min + k*(test2_records[i][:,0] - min_data)
#     test2_records[i][:,1] = config.normalization_min + k*(test2_records[i][:,1] - min_data)
#     test2_records[i][:,2] = config.normalization_min + k*(test2_records[i][:,2] - min_data)
#     test2_records[i][:,3] = config.normalization_min + k*(test2_records[i][:,3] - min_data)
#     test2_records[i][:,4] = config.normalization_min + k*(test2_records[i][:,4] - min_data)
#     test2_records[i][:,5] = config.normalization_min + k*(test2_records[i][:,5] - min_data)
#     test2_records[i][:,6] = config.normalization_min + k*(test2_records[i][:,6] - min_data)
#     test2_records[i][:,7] = config.normalization_min + k*(test2_records[i][:,7] - min_data)
#     test2_records[i][:,8] = config.normalization_min + k*(test2_records[i][:,8] - min_data)
#     test2_records[i][:,9] = config.normalization_min + k*(test2_records[i][:,9] - min_data)
#     test2_records[i][:,10] = config.normalization_min + k*(test2_records[i][:,10] - min_data)
#     test2_records[i][:,11] = config.normalization_min + k*(test2_records[i][:,11] - min_data)
#     # data.append(min(test2_records[i][:,0]))
#     #
#     # data.append(max(test2_records[i][:,1]))
#     # data.append(min(test2_records[i][:,1]))
#     #
#     # data.append(max(test2_records[i][:,2]))
#     # data.append(min(test2_records[i][:,2]))
#     #
#     # data.append(max(test2_records[i][:,3]))
#     # data.append(min(test2_records[i][:,3]))
#     #
#     # data.append(max(test2_records[i][:,4]))
#     # data.append(min(test2_records[i][:,4]))
#     #
#     # data.append(max(test2_records[i][:, 5]))
#     # data.append(min(test2_records[i][:, 5]))
#     #
#     # data.append(max(test2_records[i][:, 6]))
#     # data.append(min(test2_records[i][:, 6]))
#     #
#     # data.append(max(test2_records[i][:, 7]))
#     # data.append(min(test2_records[i][:, 7]))
#     #
#     # data.append(max(test2_records[i][:, 8]))
#     # data.append(min(test2_records[i][:, 8]))
#     #
#     # data.append(max(test2_records[i][:, 9]))
#     # data.append(min(test2_records[i][:, 9]))
#     #
#     # data.append(max(test2_records[i][:, 10]))
#     # data.append(min(test2_records[i][:, 10]))
#     #
#     # data.append(max(test2_records[i][:, 11]))
#     # data.append(min(test2_records[i][:, 11]))
#
#
# toc = time.time()
# print("all time:",toc-tic)
#

# import numpy as np
# a = np.load('ECG_test_data_normal_500record.npy')
# print(len(a))
# b = np.load('1Record_Label.npy')[0:500]
# print(len(b))
import time

import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.io as sio
import os
import config
from keras.preprocessing import sequence
import QRSDetectorOffline
import matplotlib.pyplot as plt
 # 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'
config = config.Config()
a = os.listdir(config.train_mat_path)
train_mat = [] # 存储所有6877个样本数据
for i in range(len(a)):
    if a[i].endswith('.mat'):
        train_mat.append(config.train_mat_path + a[i])

b = os.listdir(config.test_mat_path)
test_mat = [] # 存储最终的测试数据
for i in range(len(b)):
    if b[i].endswith('.mat'):
        test_mat.append(config.test_mat_path + b[i])

def data_process(all_mat):
    ECG_noqrs = []
    ECG_qrs = []
    #for recordpath in range(len(all_mat)):
    for recordpath in range(1):
        # load ECG
        mat = sio.loadmat(all_mat[recordpath])
        mat = np.array(mat['ECG']['data'][0, 0])
        mat = np.transpose(mat)  # 做转置
        signal = mat
        #print(signal.shape)
        ECG_noqrs.append(signal)

        qrsdetector = QRSDetectorOffline.QRSDetectorOffline(signal, config.sample_frequency, verbose=False,
                                                         plot_data=False, show_plot=False)
        # denoise ECG 对每一导联进行去噪 滤波
        for i in range(signal.shape[1]):
            signal[:, i] = qrsdetector.bandpass_filter(signal[:, i], lowcut=0.5, highcut=49.0,
                                                       signal_freq=config.sample_frequency, filter_order=1)

        ECG_qrs.append(signal)

    ECG_qrs = sequence.pad_sequences(ECG_qrs, maxlen=3600, dtype='float32', truncating='post')
    ECG_noqrs = sequence.pad_sequences(ECG_noqrs, maxlen=3600, dtype='float32', truncating='post')
    print(len(ECG_qrs))
    print(len(ECG_noqrs))
    print(ECG_qrs[0].shape)
    print(ECG_noqrs[0].shape)

    print(ECG_qrs[0][:,0].shape)
    print(ECG_noqrs[0][:,0].shape)
    plot_wave(ECG_qrs[0][:,0], ECG_noqrs[0][:,0])
    return ECG_qrs,ECG_noqrs

def plot_wave(ECG_qrs, ECG_noqrs):
    plt.figure()
    print(len(ECG_qrs.shape))
    print(ECG_qrs == ECG_noqrs)
    x = range(len(ECG_qrs))
    plt.plot(x, ECG_qrs, color="red")
    plt.plot(x, ECG_noqrs, color="blue")
    plt.title("qrs and no qrs wave")
    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.legend()
    plt.show()

data_process(train_mat)