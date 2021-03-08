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
test_mat = [] # 存储300个测试数据
for i in range(len(b)):
    if b[i].endswith('.mat'):
        test_mat.append(config.test_mat_path + b[i])

def data_process(all_mat):
    ECG = []
    for recordpath in range(len(all_mat)):
        # load ECG
        mat = sio.loadmat(all_mat[recordpath])
        mat = np.array(mat['ECG']['data'][0, 0])
        mat = np.transpose(mat)  # 做转置
        signal = mat

        qrsdetector = QRSDetectorOffline.QRSDetectorOffline(signal, config.sample_frequency, verbose=False,
                                                         plot_data=False, show_plot=False)
        # denoise ECG 对每一导联进行去噪
        for i in range(signal.shape[1]):
            signal[:, i] = qrsdetector.bandpass_filter(signal[:, i], lowcut=0.5, highcut=49.0,
                                                       signal_freq=config.sample_frequency, filter_order=1)

        ECG.append(signal)

    # 将所有导联的长度填充为一样的,尾部补0
    ECG = sequence.pad_sequences(ECG, maxlen=1800, dtype='float32', truncating='post')
    #np.save('ECG_train_data_process_no_wave.npy', ECG)
    #np.save('ECG_train_data_process_QRS.npy', ECG)
    #np.save('ECG_test_data_process_no_wave.npy', ECG)
    np.save('ECG_test_data_process_QRS.npy', ECG)
    return ECG

# data_process(train_mat)
data_process(test_mat)


