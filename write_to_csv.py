import config
import os
from config import Config
from os.path import basename
import csv

def write_to_csv(predict):

    config = Config()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

    b = os.listdir(config.test_mat_path)
    test_mat = []  # 存储300个测试数据
    for i in range(len(b)):
        if b[i].endswith('.mat'):
            test_mat.append(basename(b[i]).rstrip('.mat'))

    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])

        records_count = len(test_mat)
        print('record number: ', str(records_count))

        for k, label in enumerate(predict):
            result = int(label)
            record_name = test_mat[k]

            answer = [record_name, result]
            # write result
            writer.writerow(answer)

        csvfile.close()

# if __name__ == "__main__":
#     import random
#     predict = []
#     for i in range(300):
#         predict.append(random.randint(0,8))
#     qrs_detector = write_to_csv(predict)
