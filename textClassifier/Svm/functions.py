import numpy as np
import random
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# split the sample into healthy ones and ill ones
def split_sample(sample, label_list):
    healthy = []
    ill = []
    for i in range(sample.shape[0]):
        label = label_list[i]
        value = sample[i]
        # 0:healthy
        if label == 0:
            healthy.append(value)
        elif label == 1:
            ill.append(value)
    healthy = np.array(healthy)
    ill = np.array(ill)
    return healthy, ill


# listTemp 为列表 平分后每份列表的的个数n
def split_list(x, n, newList=[]):
    if len(x) <= n:
        newList.append(x)
        return newList
    else:
        newList.append(x[:n])
        return split_list(x[n:], n)


# construct train data and test data
# 10sets:each set do not cross
def get_10sets(healthy, ill):
    h_train, h_test = train_test_split(healthy, train_size=3000, random_state=1)
    i_train, i_test = train_test_split(ill, train_size=300, random_state=1)
    train1 = []
    train2 = []
    train3 = []
    train4 = []
    train5 = []
    train6 = []
    train7 = []
    train8 = []
    train9 = []
    train10 = []

    # label_train
    label_train = [0] * 300 + [1] * 300
    # randnum = 2020

    train = split_list(h_train, 300)
    # train1
    for i in range(len(train[0])):
        value = train[0][i]
        train1.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train1.append(value)
    # random.seed(randnum)
    # random.shuffle(train1)
    # label1 = label_train
    # random.seed(randnum)
    # random.shuffle(label1)
    # train2
    for i in range(len(train[1])):
        value = train[1][i]
        train2.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train2.append(value)
    # random.seed(randnum)
    # random.shuffle(train2)
    # label2 = label_train
    # random.seed(randnum)
    # random.shuffle(label2)
    # train3
    for i in range(len(train[2])):
        value = train[2][i]
        train3.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train3.append(value)
    # random.seed(randnum)
    # random.shuffle(train3)
    # label3 = label_train
    # random.seed(randnum)
    # random.shuffle(label3)
    # train4
    for i in range(len(train[3])):
        value = train[3][i]
        train4.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train4.append(value)
    # random.seed(randnum)
    # random.shuffle(train4)
    # label4 = label_train
    # random.seed(randnum)
    # random.shuffle(label4)
    # train5
    for i in range(len(train[4])):
        value = train[4][i]
        train5.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train5.append(value)
    # random.seed(randnum)
    # random.shuffle(train5)
    # label5 = label_train
    # random.seed(randnum)
    # random.shuffle(label5)
    # train6
    for i in range(len(train[5])):
        value = train[5][i]
        train6.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train6.append(value)
    # random.seed(randnum)
    # random.shuffle(train6)
    # label6 = label_train
    # random.seed(randnum)
    # random.shuffle(label6)
    # train7
    for i in range(len(train[6])):
        value = train[6][i]
        train7.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train7.append(value)
    # random.seed(randnum)
    # random.shuffle(train7)
    # label7 = label_train
    # random.seed(randnum)
    # random.shuffle(label7)
    # train8
    for i in range(len(train[7])):
        value = train[7][i]
        train8.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train8.append(value)
    # random.seed(randnum)
    # random.shuffle(train8)
    # label8 = label_train
    # random.seed(randnum)
    # random.shuffle(label8)
    # train9
    for i in range(len(train[8])):
        value = train[8][i]
        train9.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train9.append(value)
    # random.seed(randnum)
    # random.shuffle(train9)
    # label9 = label_train
    # random.seed(randnum)
    # random.shuffle(label9)
    # train10
    for i in range(len(train[9])):
        value = train[9][i]
        train10.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train10.append(value)
    # random.seed(randnum)
    # random.shuffle(train10)
    # label10 = label_train
    # random.seed(randnum)
    # random.shuffle(label10)
    train_set = []
    train_set.append(train1)
    train_set.append(train2)
    train_set.append(train3)
    train_set.append(train4)
    train_set.append(train5)
    train_set.append(train6)
    train_set.append(train7)
    train_set.append(train8)
    train_set.append(train9)
    train_set.append(train10)
    # print(len(train_set))
    # label_set = []
    # label_set.append(label1)
    # label_set.append(label2)
    # label_set.append(label3)
    # label_set.append(label4)
    # label_set.append(label5)
    # label_set.append(label6)
    # label_set.append(label7)
    # label_set.append(label8)
    # label_set.append(label9)
    # label_set.append(label10)

    # 测试集
    test = []
    for i in range(len(h_test)):
        value = h_test[i]
        test.append(value)
    for i in range(len(i_test)):
        value = i_test[i]
        test.append(value)
    test = np.array(test)
    # random.seed(randnum)
    # random.shuffle(test)

    # label_test
    label_test = [0] * len(h_test) + [1] * len(i_test)
    label_test = np.array(label_test)
    # random.seed(randnum)
    # random.shuffle(label_test)

    # return train_set, label_set, test, label_test
    return train_set, label_train, test, label_test


def get_15sets(healthy, ill):
    h_train, h_test = train_test_split(healthy, train_size=3000, random_state=1)
    i_train, i_test = train_test_split(ill, train_size=300, random_state=1)

    # create train sets
    classifier_num = 15
    size = 200
    x_train = []
    y_train = []
    label_train = [0] * 200 + [1] * 200
    # randnum = 2020

    for i in range(classifier_num):
        x_train.append([])
        # label_i = label_train
        for j in range(size):
            idx = random.randint(0, len(h_train) - 1)
            x_train[i].append(h_train[idx])
        for n in range(size):
            idx = random.randint(0, len(i_train) - 1)
            x_train[i].append(i_train[idx])
        y_train.append(label_train)
        # random.seed(randnum)
        # random.shuffle(x_train[i])
        # random.seed(randnum)
        # random.shuffle(label_i)
        # y_train.append(label_i)
    print(len(x_train))
    print(len(y_train))
    # 测试集
    test = []
    for i in range(len(h_test)):
        value = h_test[i]
        test.append(value)
    for i in range(len(i_test)):
        value = i_test[i]
        test.append(value)
    test = np.array(test)
    # random.seed(randnum)
    # random.shuffle(test)

    # label_test
    label_test = [0] * len(h_test) + [1] * len(i_test)
    label_test = np.array(label_test)
    # random.seed(randnum)
    # random.shuffle(label_test)

    return x_train, y_train, test, label_test


def get_sets(healthy, ill):
    h_train, h_test = train_test_split(healthy, train_size=3000, random_state=1)
    i_train, i_test = train_test_split(ill, train_size=300, random_state=1)

    label_train = [0] * 3000 + [1] * 300
    label_train = np.array(label_train)
    # label_test
    label_test = [0] * len(h_test) + [1] * len(i_test)
    label_test = np.array(label_test)

    train = []
    for i in range(len(h_train)):
        value = h_train[i]
        train.append(value)
    for i in range(len(i_train)):
        value = i_train[i]
        train.append(value)
    train = np.array(train)
    # randnum = 2020
    # random.seed(randnum)
    # random.shuffle(train)
    # random.seed(randnum)
    # random.shuffle(label_train)

    # 测试集
    test = []
    for i in range(len(h_test)):
        value = h_test[i]
        test.append(value)
    for i in range(len(i_test)):
        value = i_test[i]
        test.append(value)
    test = np.array(test)
    # random.seed(randnum)
    # random.shuffle(test)
    # random.seed(randnum)
    # random.shuffle(label_test)

    return train, label_train, test, label_test


# adjust parameter
def get_best_parameter(X_train, y_train):
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # scores = ['precision', 'recall']
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]
    scores = ['precision']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        # 用训练集训练这个学习器 clf
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)
        # print()
        # print("Grid scores on development set:")
        # print()
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        #
        # # 看一下具体的参数间不同数值的组合后得到的分数是多少
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()
    return clf


# voting
def voting(predict_result):
    result = []
    for i in range(predict_result.shape[1]):
        h_num = 0
        i_num = 0
        for j in range(predict_result.shape[0]):
            value = predict_result[j][i]
            if value == 0:
                h_num = h_num + 1
            elif value == 1:
                i_num = i_num + 1
        if h_num > i_num:
            result.append(0)
        else:
            result.append(1)
    return result


def assess_model(y_true, y_predict):
    # accuracy 准确率
    accuracy = metrics.accuracy_score(y_true, y_predict)
    print('Accuracy:', accuracy)
    # Precision 精确度
    p = metrics.precision_score(y_true, y_predict)
    print('Precision:', p)

    # recall
    recall = metrics.recall_score(y_true, y_predict)
    print('Recall:', recall)

    # F1_score
    f1_macro = metrics.f1_score(y_true, y_predict)
    print('F1_score: {0}'.format(f1_macro))
    f1_micro = metrics.f1_score(y_true, y_predict, average='micro')
    f1_macro = metrics.f1_score(y_true, y_predict, average='macro')
    print('f1_micro: {0}'.format(f1_micro))
    print('f1_macro: {0}'.format(f1_macro))


# confusion matrix
def plot_matrix(y_true, y_pred, labels_name, title=None, axis_labels=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    print(cm)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例
    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    thresh = cm.max()/2
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            plt.text(j, i, format(cm[i][j], 'd'),
                    ha ="center", va="center",
                    color ="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    plt.show()





