import time

import numpy as np
import scipy.io as sio
import os
import config
from keras.preprocessing import sequence
import QRSDetectorOffline

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
    ECG = []
    for recordpath in range(len(all_mat)):
    #for recordpath in range(1):
        # load ECG
        mat = sio.loadmat(all_mat[recordpath])
        mat = np.array(mat['ECG']['data'][0, 0])
        mat = np.transpose(mat)  # 做转置
        signal = mat
        #print(signal.shape)

        qrsdetector = QRSDetectorOffline.QRSDetectorOffline(signal, config.sample_frequency, verbose=False,
                                                         plot_data=False, show_plot=False)
        # denoise ECG 对每一导联进行去噪 滤波
        for i in range(signal.shape[1]):
            signal[:, i] = qrsdetector.bandpass_filter(signal[:, i], lowcut=0.5, highcut=49.0,
                                                       signal_freq=config.sample_frequency, filter_order=1)

        ECG.append(signal)
        # print(ECG[0].shape)
        # print(ECG[0])
        # print(signal)
    # 将所有导联的长度填充为一样的,尾部补0
    ECG = sequence.pad_sequences(ECG, maxlen=3600, dtype='float32', truncating='post')
    print(len(ECG))
    calculate_max_min(ECG)

    #np.save('ECG_train_data_process_no_wave.npy', ECG)
    # np.save('ECG_train_data_process_3600QRS.npy', ECG)
    #np.save('ECG_test_data_process_no_wave.npy', ECG)
    # np.save('ECG_test_data_process_3600QRS.npy', ECG)
    return ECG

def calculate_max_min(ECG):
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
    normalization(ECG, config.max_data, config.min_data)
    print(max(data))
    print(min(data))
    toc = time.time()
    print("data normalization takes time:", toc - tic)
    return max_data,min_data

# 数据归一化到指定区间
def normalization(ECG, max_data, min_data):
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
    np.save('ECG_test_data_normal_500record.npy', ECG)
    return ECG

#data_process(train_mat)
data_process(test_mat)


