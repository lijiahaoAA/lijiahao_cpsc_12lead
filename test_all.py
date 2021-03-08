import numpy as np

a = np.load('ECG_train_data_process_QRS.npy')
b = np.load('ECG_test_data_process_QRS.npy')
print(a.shape)
print(b.shape)