import random
import os
import argparse
import csv
import glob

from keras_preprocessing import sequence
from keras.models import load_model
import numpy as np
import scipy.io as sio
from os.path import basename

'''
cspc2018_challenge score
Written by:  Xingyao Wang, Feifei Liu, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
'''

'''
Save prdiction answers to answers.csv in local path, the first column is recording name and the second
column is prediction label, for example:
Recoding    Result
B0001       1
.           .
.           .
.           .
'''
def cpsc2018(record_base_path):
    # ecg = scipy.io.loadmat(record_path)
    ###########################INFERENCE PART################################

    def predict_test(recordpaths):
        model = load_model('lijiahao_Net_QRS.h5')
        # prepare input
        Input = data_process(recordpaths)
        results = model.predict(Input)
        return results.argmax(1)

    def data_process(all_mat):
        ECG = []
        for recordpath in range(len(all_mat)):
            # load ECG
            mat = sio.loadmat(all_mat[recordpath])
            mat = np.array(mat['ECG']['data'][0, 0])
            mat = np.transpose(mat)  # 做转置
            signal = mat
            ECG.append(signal)
        # 将所有导联的长度填充为一样的
        ECG = sequence.pad_sequences(ECG, maxlen=1800, dtype='float32', truncating='post')
        return ECG

    ## Please process the ecg data, and output the classification result.
    ## result should be an integer number in [1, 9].
    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])
        recordpaths = glob.glob(os.path.join(record_base_path, '*.mat'))
        print(recordpaths)
        records_count = len(recordpaths)
        print('record number: ', str(records_count))
        labels = predict_test(recordpaths)
        for k, label in enumerate(labels):
            result = int(label + 1)
            record_name = basename(recordpaths[k]).rstrip('.mat')
            # print(record_name)
            answer = [record_name, result]
            # write result
            writer.writerow(answer)

        csvfile.close()

    ###########################INFERENCE PART################################

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-p',
#                         '--recording_path',
#                         help='path saving test record file')
#
#     args = parser.parse_args()
#
#     result = cpsc2018(record_base_path=args.recording_path)
record_base_path = 'D:/data-set/12leadsECG/validation_set/'
cpsc2018(record_base_path)