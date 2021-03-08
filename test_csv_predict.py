import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import csv
import os
from os.path import basename

mat_name = os.listdir('D:/data-set/12leadsECG/validation_set/')
mat_real_name = [] #300个文件名称
for i in mat_name:
    if i.endswith('.mat'):
        mat_real_name.append(i)

model = load_model('lijiahao_Net.h5')
# test_records = np.load('ECG_test_data_process_no_wave.npy')
test_records = np.load('ECG_test_data_process_QRS.npy')
test_label = np.load('1Record_Label.npy')[0:300]-1
test_one_hot = np_utils.to_categorical(test_label, num_classes=9)
print(test_records.shape)
print(test_label.shape)
print(test_one_hot.shape)
scores_Test = model.evaluate(test_records,test_one_hot)
print("测试集精确度："+str(scores_Test[1]))

predict = model.predict(test_records)
predict = np.argmax(predict,axis=1)+1
print(predict)

with open('answers.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    # column name
    writer.writerow(['Recording', 'Result'])
    for k, label in enumerate(predict):

        record_name =mat_real_name[k].rstrip('.mat')
        # print(record_name)
        answer = [record_name, label]
        # write result
        writer.writerow(answer)

    csvfile.close()
