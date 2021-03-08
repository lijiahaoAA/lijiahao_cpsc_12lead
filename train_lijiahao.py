import os
import time

import numpy as np

from scipy.signal import resample
from sklearn.preprocessing import scale
from keras.utils import np_utils
from keras.models import Model,load_model
from keras.layers import Conv1D,MaxPool1D,Add,Activation,Dropout,BatchNormalization,Input,Lambda,GlobalAveragePooling1D,Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import pandas as pd
import config
import warnings
import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import data_process
from keras.models import load_model
import data_process

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'
warnings.filterwarnings("ignore")
seed = np.random.seed(12)
config = config.Config()
input_shape = [1800,12]

tic = time.time()
# ==================================工具方法=======================================
def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=2)

def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)

def lr_schedule(epoch):
    lr = 0.1
    if epoch >= 20 and epoch < 40:
        lr = 0.01
    if epoch >= 40:
        lr = 0.001
    print('Learning rate: ', lr)
    return lr
# ==================================数据载入和整理==================================
target_sig_length = input_shape[0] # 输入信号的长度 1280
tic = time.time()

# train = np.load('ECG_train_data_process_no_wave.npy')
train = np.load('ECG_train_data_process_QRS.npy')
label = np.load('1Record_Label.npy')-1

train_records, val_records, train_labels, val_labels = train_test_split(
    train, label, test_size=0.2, random_state=seed)

# 将标签数据转换为one-hot数据
train_labels = np_utils.to_categorical(train_labels, num_classes=9)
val_labels = np_utils.to_categorical(val_labels, num_classes=9)

# train_records = train_records[0:500]
# train_labels = train_labels[0:500]
# val_records = val_records[0:300]
# val_labels = val_labels[0:300]
# print(train_records.shape)
# print(train_labels.shape)
# print(val_records.shape)
# print(val_labels.shape)
toc = time.time()

print('Time for data processing--- '+str(toc-tic)+' seconds---')

# =====================================神经网络的生成与搭建==================================
def nn_network():

    inputs = Input(shape=input_shape, dtype='float32')
    layer = Conv1D(filters=12, kernel_size=32, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    shortcut = MaxPool1D(pool_size=1)(layer)
    layer = Conv1D(filters=12, kernel_size=32, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=12, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=12, kernel_size=32, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=12, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=1)(layer)
    shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=24, kernel_size=32, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=24, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=24, kernel_size=32, strides=2, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=24, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=1)(layer)
    shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=48, kernel_size=32, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=48, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=48, kernel_size=32, strides=2, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=48, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=1)(layer)
    shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=96, kernel_size=32, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=96, kernel_size=32, strides=1, padding='same')(layer)
    layer = Add()([shortcut, layer])

    shortcut = MaxPool1D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv1D(filters=96, kernel_size=32, strides=2, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Conv1D(filters=96, kernel_size=32, strides=1, padding='same')(layer)
    #layer = Add()([shortcut, layer])

    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Dense(9)(layer)

    output = Activation('softmax')(layer)
    model = Model(inputs=[inputs], outputs=[output])
    return model

model = nn_network()
print(model.summary())
optimizer = SGD(lr=lr_schedule(0), momentum=config.momentum)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
model_name = 'lijiahao_Net_QRS.h5'
checkpoint = ModelCheckpoint(filepath=model_name,
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')

lr_scheduler = LearningRateScheduler(lr_schedule)
callback_lists = [checkpoint, lr_scheduler]
H = model.fit(x=train_records, y=train_labels, batch_size=32, epochs=config.epoch,
          verbose=1, validation_data=(val_records,val_labels), callbacks=callback_lists) # verbose=1表示输出进度条记录

# ======================================测试集上评估===============================================
# 训练好的模型来评估测试集
from keras.models import load_model
model = load_model('lijiahao_Net_QRS.h5')
# test_records = np.load('ECG_test_data_process_no_wave.npy')
test_records = np.load('ECG_test_data_process_QRS.npy')
test_label = np.load('1Record_Label.npy')[0:300]
# scores_Test = model.evaluate(TestX,TestY)
# print("测试集精确度："+str(scores_Test[1]))

predict = model.predict(test_records)
predict = np.argmax(predict,axis=1)+1
print(predict)
cross_mutrix = pd.crosstab(test_label, predict, rownames=['labels'], colnames=['predict'])
print(cross_mutrix)

toc = time.time()
print("总用时："+str(toc-tic))

# # ==========================================绘图==================================================
N = np.arange(0, config.epoch)
plt.figure()
plt.subplot(121)
plt.plot(N, H.history["loss"], label="train_loss",color="red")
plt.plot(N, H.history["val_loss"], label="val_loss",color="blue")
plt.title("Training Loss in Train and Val Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(N, H.history["categorical_accuracy"], label="categorical_accuracy",color="red")
plt.plot(N, H.history["val_categorical_accuracy"], label="val_categorical_accuracy",color="blue")
plt.title("Training Accuracy in Train and Val Set")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()

plt.suptitle("Loss and Accuracy")
plt.show()
