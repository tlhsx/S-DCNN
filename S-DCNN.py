from keras.models import Sequential
from keras.layers import MaxPooling1D, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution1D as Conv1D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
# from keras import backend as K
import keras_metrics as km

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import pandas as pd
import numpy as np
from math import sqrt
from utils import *
from sklearn.metrics import roc_auc_score
import csv
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import math
from sklearn import preprocessing
from keras import regularizers
from keras.layers.normalization import BatchNormalization


for CELL_SIZE in [16]:
    for BATCH_SIZE in [32]:
        def get_callbacks(patience=5):
            es = EarlyStopping('val_loss', patience=patience, mode="min")
            return [es]


        def create_model(x_train, y_train, x_test, y_test, file, params=None):
            if 'run' in file:
                input_shape = (250, 1)
            if '锰' in file:
                input_shape = (5, 14, 1)

            model = Sequential()

            # Conv Layer 1
            #cnn
            model.add(Conv1D(CELL_SIZE, kernel_size=3,   padding='same', input_shape=input_shape))#activation='relu',
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            #dnn
            #model.add(Dense(CELL_SIZE, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu', input_shape=input_shape))
            #model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(MaxPooling1D(pool_size=2))
            #model.add(Dropout(0.4))

            # Conv Layer 2
            model.add(Conv1D(CELL_SIZE,  kernel_size=3))#, activation='relu'
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            #model.add(Dense(CELL_SIZE, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),  activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            #model.add(Dropout(0.4))

            # Conv Layer 3
            model.add(Conv1D(CELL_SIZE,  kernel_size=3))#, activation='relu'
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            #model.add(Dense(CELL_SIZE,  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            #model.add(Dropout(0.4))

            # Conv Layer 4
            #model.add(Conv1D(CELL_SIZE, kernel_size=2, activation='relu'))
            #model.add(Conv1D(CELL_SIZE, kernel_regularizer=regularizers.l2(0.001), kernel_size=2, activation='relu'))
            #model.add(Dense(CELL_SIZE, activation='relu'))
            #model.add(MaxPooling1D(pool_size=2))
            #model.add(Dropout(0.2))

            # Conv Layer 5
            #model.add(Conv1D(16,kernel_size=2, activation='relu'))
            #model.add(Dense(16, activation='relu'))
            # model.add(MaxPooling1D(pool_size=1))
            # model.add(Dropout(0.2))

            # Conv Layer 6
            #model.add(Conv1D(8,kernel_size=2, activation='relu'))
            # model.add(MaxPooling1D(pool_size=1))
            # model.add(Dropout(0.2))

            # Conv Layer 7
            #model.add(Conv1D(512,kernel_size=2, activation='relu'))
            # model.add(MaxPooling1D(pool_size=1))
            # model.add(Dropout(0.2))
            # Flatten the data for upcoming dense layers
            model.add(Flatten())

            # Dense Layers 1
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))

            # Dense Layers 2
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))

            # Dense Layers 3
            #model.add(Dense(16, activation='relu'))
            #model.add(Dropout(0.2))

            # Dense Layers 4
            #model.add(Dense(16, activation='relu'))
            #model.add(Dropout(0.2))

            # Output Layer
            model.add(Dense(2, activation='softmax'))#sigmoid

            #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            optimizer = SGD(lr=0.05)
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['acc', km.sparse_categorical_f1_score(),
                                   km.sparse_categorical_precision(), km.sparse_categorical_recall()])
            model.summary()

            # callbacks = get_callbacks(patience=10)
            history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=60, verbose=1,
                                validation_data=(x_test, y_test) )#shuffle=True

            return history, model


        def read_data(n_fold, file='run.txt', type='cnn'):
            if 'run' in file:
                sep = '\t'
            if '锰' in file:
                sep = ' '
            data = pd.read_table(file, sep=sep,header=None)
            feature_set = data.iloc[:, :-1]
            feature_set = feature_set.values
            # Max-Min标准化
            # 建立MinMaxScaler对象
            #minmax = preprocessing.MinMaxScaler()
            # 标准化处理
            #feature_set = minmax.fit_transform(feature_set)
            label_set = data.iloc[:, -1]
            label_set = LabelEncoder().fit_transform(label_set)
            smo = SMOTE(random_state=0, sampling_strategy=1)#, sampling_strategy=1
            feature_set_SMOTED, label_set_SMOTED = smo.fit_resample(feature_set, label_set)
            #print(feature_set_SMOTED.isnull().any())
            #print(feature_set_SMOTED.isin([np.nan, np.inf, -np.inf]).any(1).sum())
            #rus = RandomUnderSampler(random_state=0, sampling_strategy=1)
            #feature_set_SMOTED, label_set_SMOTED = rus.fit_resample(feature_set, label_set)
            if type == 'rnn':
                #feature_set = feature_set
                feature_set_SMOTED = feature_set_SMOTED
                test_size = 0.5
                random_state = 42
            elif type == 'cnn':
                if 'run' in file:
                    #feature_set = feature_set.reshape((feature_set.shape[0],242, 1))
                    feature_set_SMOTED = feature_set_SMOTED.reshape((feature_set_SMOTED.shape[0], 250, 1))
                    test_size = 0.5
                    random_state = 42
                if '锰' in file:
                    feature_set = feature_set.reshape((feature_set.shape[0], 5, 14, 1))
                    test_size = 0.5
                    random_state = 42

            skf = StratifiedKFold(n_splits=n_fold)
            #for train_index, test_index in skf.split(feature_set, label_set):
                #X_train, X_test, y_train, y_test = feature_set[train_index], feature_set[test_index], \
                 #                                  label_set[train_index], label_set[test_index]
            for train_index, test_index in skf.split(feature_set_SMOTED, label_set_SMOTED):
                X_train, X_test, y_train, y_test = feature_set_SMOTED[train_index], feature_set_SMOTED[test_index], \
                                                    label_set_SMOTED[train_index], label_set_SMOTED[test_index]
            # X_train, X_test, y_train, y_test = train_test_split(feature_set, label_set,
            #                                                     test_size=test_size, random_state=random_state)
                yield X_train, X_test, y_train, y_test



        file = 'run.txt'
        n_fold = 5
        result = pd.DataFrame(index=["loss", "accuracy", "F1", "Precision", "Recall", "Sensitivity", "Specificity", "MCC", "AUC"],
                              columns=['1_fold', '2_fold', '3_fold', '4_fold', '5_fold'])

        for i in range(n_fold):
            X_train, X_test, y_train, y_test = next(read_data(n_fold, file=file))
            history, model = create_model(X_train, y_train, X_test, y_test, file)
            score = model.evaluate(X_test, y_test, verbose=1)
            proba = model.predict_proba(X_test)[:, 1]
            pred = model.predict_classes(X_test)
            con_mat = confusion_matrix(y_test, pred)

            # print('Test loss:', score[0])
            # print('Test accuracy:', score[1])
            # print('Test F1:', score[2])
            # print('Test Precision:', score[3])
            # print('Test Recall:', score[4])
            # print('Test Sensitivity:', con_mat[0, 0] / np.sum(con_mat, axis=1)[0])
            # print('Test Specificity:', con_mat[1, 1] / np.sum(con_mat, axis=1)[1])
            # print('Test MCC:', np.linalg.det(con_mat) /
            #       sqrt(np.sum(con_mat, axis=1).cumprod()[-1] * np.sum(con_mat, axis=0).cumprod()[-1]))
            plot(history)
            auc = plot_roc(y_test, proba)
            #auc = roc_auc_score(y_test, proba)
            result.loc[:, "%d_fold" % (i+1)] = score + [con_mat[1, 1] / np.sum(con_mat, axis=1)[1],
                                                        con_mat[0, 0] / np.sum(con_mat, axis=1)[0],
                                                        np.linalg.det(con_mat) / sqrt(np.sum(con_mat, axis=1).cumprod()[-1] *
                                                                                  np.sum(con_mat, axis=0).cumprod()[-1]),
                                                        auc]
            # print('AUC Score:', auc)
            print('Confusion Matrix:\n', con_mat)
            with open("data3.csv", "a+", encoding="utf-8", newline="") as f:
                f.write(str(con_mat))
                f.write("\n")
        print(result)
        result.to_csv('cnn_result_3_%d_%d.csv' % (CELL_SIZE, BATCH_SIZE), index=True)
        model.save('cnn_model_3_%d_%d.h5' % (CELL_SIZE, BATCH_SIZE))

