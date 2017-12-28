# -*- coding:utf-8 -*-
import arff
import warnings
import numpy as np
import pandas as pd
from math import log
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def naive_Bayes(Features,labels,Predict):
    set_class = []
    for i in labels:
        if i not in set_class:
            set_class.append(i)
    class_num  =len(set_class)
    Features_rows_num ,Features_cols_num = Features.shape
    pro_of_classes  =[0]*class_num
    for (i,label) in enumerate(set_class):
        prob_class =len(labels[labels ==label])/Features_rows_num
        for (j,cols_of_samples) in enumerate(Predict):
            Features_of_classes =Features[labels ==label]
            num_count = 1e-10
            for cols_features_num in Features_of_classes[:,j]:
                 if cols_features_num ==cols_of_samples:
                        num_count =+1
            pro_of_classes[i]+= log(num_count/len(Features_of_classes))
        pro_of_classes[i]+=log(prob_class)
    max_index =np.argmax(pro_of_classes)
    return set_class[int(max_index)]

if __name__ == '__main__':
    train_size = 0.8
    name = input('Please input the dataset name: weather or soybean: ')
    data = arff.load(open('%s.arff' % name, 'r'))
    data = np.array(pd.DataFrame(data['data']).dropna())
    label_encoder = LabelEncoder()
    encoded = np.zeros(data.shape)
    for i in range(len(data[0])):
        encoded[:, i] = label_encoder.fit_transform(data[:, i])
    x, y = encoded[:, :-1], encoded[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=train_size)
    output = []
    for i in x_test:
        output.append(naive_Bayes(x_train, y_train, i))
    accuracy = accuracy_score(y_test, output) * 100
    print('\n%s test accuracy: %.2f%%' % (name, accuracy),
          '\nTrue:\n', y_test, '\nPredict：\n', np.array(output))
