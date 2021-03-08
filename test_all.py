import time

import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.io as sio
import os
import config
from keras.preprocessing import sequence
import QRSDetectorOffline
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

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
    ECG_1 = []
    ECG_2 = []
    ECG_3 = []
    #for recordpath in range(len(all_mat)):
    for recordpath in range(1):
        # load ECG
        mat = sio.loadmat(all_mat[recordpath])
        mat = np.array(mat['ECG']['data'][0, 0])
        mat = np.transpose(mat)  # 做转置
        signal = mat
        ECG_1.append(signal)
        #print(signal.shape)

        qrsdetector = QRSDetectorOffline.QRSDetectorOffline(signal, config.sample_frequency, verbose=False,
                                                         plot_data=False, show_plot=False)
        # denoise ECG 对每一导联进行去噪 滤波
        for i in range(signal.shape[1]):
            signal[:, i] = qrsdetector.bandpass_filter(signal[:, i], lowcut=0.5, highcut=49.0,
                                                       signal_freq=config.sample_frequency, filter_order=1)

        ECG_2.append(signal)
        # print(ECG[0].shape)
        # print(ECG[0])
        # print(signal)
    # 将所有导联的长度填充为一样的,尾部补0
    # ECG_1 = sequence.pad_sequences(ECG_1, maxlen=3600, dtype='float32', truncating='post')
    ECG_2 = sequence.pad_sequences(ECG_2, maxlen=3600, dtype='float32', truncating='post')
    print(len(ECG_1))
    print(len(ECG_2))
    # plot_wave(ECG_1[0][:,0],ECG_2[0][:,0])
    calculate_max_min(ECG_2,ECG_1[0][:,0],ECG_2[0][:,0])

    #np.save('ECG_train_data_process_no_wave.npy', ECG)
    # np.save('ECG_train_data_process_3600QRS.npy', ECG)
    #np.save('ECG_test_data_process_no_wave.npy', ECG)
    # np.save('ECG_test_data_process_3600QRS.npy', ECG)
    return ECG_1, ECG_2

def calculate_max_min(ECG,ECG_1,ECG_2):
    data = []
    tic = time.time()
    for i in range(len(ECG)):
        data.append(max(ECG[i][:, 0]))
        data.append(min(ECG[i][:, 0]))

        data.append(max(ECG[i][:, 1]))
        data.append(min(ECG[i][:, 1]))

        data.append(max(ECG[i][:, 2]))
        data.append(min(ECG[i][:, 2]))

        data.append(max(ECG[i][:, 3]))
        data.append(min(ECG[i][:, 3]))

        data.append(max(ECG[i][:, 4]))
        data.append(min(ECG[i][:, 4]))

        data.append(max(ECG[i][:, 5]))
        data.append(min(ECG[i][:, 5]))

        data.append(max(ECG[i][:, 6]))
        data.append(min(ECG[i][:, 6]))

        data.append(max(ECG[i][:, 7]))
        data.append(min(ECG[i][:, 7]))

        data.append(max(ECG[i][:, 8]))
        data.append(min(ECG[i][:, 8]))

        data.append(max(ECG[i][:, 9]))
        data.append(min(ECG[i][:, 9]))

        data.append(max(ECG[i][:, 10]))
        data.append(min(ECG[i][:, 10]))

        data.append(max(ECG[i][:, 11]))
        data.append(min(ECG[i][:, 11]))

    # print(len(data))
    with open("2.txt", 'w') as file:
        data1 = str(data)
        file.write(data1)
        file.close()
    max_data = max(data) # 训练集和测试集中在归一化到某个范围内时需要保证这个max_data和min_data是一致的
    min_data = min(data)
    normalization(ECG, config.max_data, config.min_data, ECG_1, ECG_2)
    print(max(data))
    print(min(data))
    toc = time.time()
    print("data normalization takes time:", toc - tic)
    return max_data,min_data

# 数据归一化到指定区间
def normalization(ECG, max_data, min_data, ECG_1, ECG_2):
    if(max_data - min_data == 0):
        print("分母为零，请检查")
        return
    k = (config.normalization_max - config.normalization_min)/((max_data - min_data) * 1.0) # 比例系数
    for i in range(len(ECG)):
        ECG[i][:, 0] = config.normalization_min + k * (ECG[i][:, 0] - min_data)
        ECG[i][:, 1] = config.normalization_min + k * (ECG[i][:, 1] - min_data)
        ECG[i][:, 2] = config.normalization_min + k * (ECG[i][:, 2] - min_data)
        ECG[i][:, 3] = config.normalization_min + k * (ECG[i][:, 3] - min_data)
        ECG[i][:, 4] = config.normalization_min + k * (ECG[i][:, 4] - min_data)
        ECG[i][:, 5] = config.normalization_min + k * (ECG[i][:, 5] - min_data)
        ECG[i][:, 6] = config.normalization_min + k * (ECG[i][:, 6] - min_data)
        ECG[i][:, 7] = config.normalization_min + k * (ECG[i][:, 7] - min_data)
        ECG[i][:, 8] = config.normalization_min + k * (ECG[i][:, 8] - min_data)
        ECG[i][:, 9] = config.normalization_min + k * (ECG[i][:, 9] - min_data)
        ECG[i][:, 10] = config.normalization_min + k * (ECG[i][:, 10] - min_data)
        ECG[i][:, 11] = config.normalization_min + k * (ECG[i][:, 11] - min_data)

    # np.save('ECG_train_data_normal.npy', ECG)
    # np.save('ECG_test_data_normal_500record.npy', ECG)
    plot_wave(ECG_1,ECG_2,ECG[0][:,0])
    return ECG

def plot_wave(ECG_qrs, ECG_noqrs, ECG_3):
    plt.figure()
    print(len(ECG_qrs.shape))
    print(len(ECG_noqrs.shape))
    print(len(ECG_3.shape))

    plt.plot(range(3600), ECG_qrs[0:3600], color="red",label="去噪数据")
    # .plot(range(3600), ECG_noqrs, color="blue")

    plt.plot(range(3600), ECG_3, color="blue", label="归一化数据")
    plt.title("去噪数据波形对比归一化到[-3,3]数据波形")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend(loc="best")
    plt.show()

#data_process(train_mat)
data_process(test_mat)




