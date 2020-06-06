import numpy as np
from textClassifier.Svm import functions
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, roc_curve,  \
            classification_report, recall_score, precision_score

# load sample vectors and label
sample_vec = np.load('../data/input/doc2vec_sample_vector.npy')
label_list = np.load('../data/input/label.npy')
label_list = list(label_list)

# split sample into healthy ones and ill ones
healthy, ill = functions.split_sample(sample_vec, label_list)
train_set, label_train, test, label_test = functions.get_15sets(healthy, ill)

# create svm model
model_num = 15
svm_model = []
for i in range(model_num):
    # 调参
    print('model:  '+str(i))
    x_train = train_set[i]
    y_train = label_train[i]
    # min_max_scaler = preprocessing.MinMaxScaler()
    # scaler.append(min_max_scaler)
    # x_train = min_max_scaler.fit_transform(X_train)
    print(x_train)
    clf = functions.get_best_parameter(x_train, y_train)
    svm_model.append(clf)

# test sets
test = test
# label_test = label_train

# test models
predict_result = []
for i in range(model_num):
    # x_test = scaler[i].fit_transform(test)
    y_predict = svm_model[i].predict(test)
    predict_result.append(y_predict)
predict_result = np.array(predict_result)

# get the predicted result by voting
result = functions.voting(predict_result)

# assess the model by accuracy, precision, recall, f1-score, confusion_matrix
y_true = label_test
y_predict = result
# functions.assess_model(y_true, y_predict)
# functions.plot_matrix(y_true, y_predict, [0, 1], title='confusion_matrix_randSVM_15',
#             axis_labels=['healthy', 'ill'])

print('1. The F-1 score of the model {}\n'.format(f1_score(y_true, y_predict, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_true, y_predict, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_true, y_predict)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true, y_predict)))
print('5. The precision of the model {}\n'.format(precision_score(y_true, y_predict)))


