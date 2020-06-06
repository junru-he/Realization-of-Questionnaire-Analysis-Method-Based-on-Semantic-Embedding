import logging
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from sklearn.metrics import f1_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, GRU
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.layers import Embedding, Dense
from keras.layers import Concatenate, SpatialDropout1D

from keras.models import *
from keras.callbacks import *
import os
import time
import gc
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()


#设置随机种子保证可重复性
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything()


# 划分训练集、测试集
def read_data():
    print('Reading data......')
    label = np.load('../data/input/label.npy')
    data = []

    with open('../data/input/non_seq_document.seq') as f:
        for line in f.readlines():
            print(line)
            line = line.strip()
            data.append(line)
    print(label.shape)
    print(len(data))
    return data, label


data, label = read_data()
X_train, X_test, Y_train, Y_true = train_test_split(data,label,test_size=0.2, random_state=0, stratify=label)
print(len(X_train), len(X_test), len(Y_train), len(Y_true))
print(X_train[0])

train_X, val_X, train_y, val_y = train_test_split(X_train,Y_train,test_size=0.2, random_state=0, stratify=Y_train)
print(len(train_X), len(val_X), len(train_y), len(val_y))

# 超参数
embed_size = 100 # how big is each word vector
max_features = 162 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 55 # max number of words in a question to use #99.99%

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features, filters='')
tokenizer.fit_on_texts(data)

train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
X_test = tokenizer.texts_to_sequences(X_test)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

del data, label
gc.collect()


def embedding_w2v():
    print('running embedding_w2c......')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec.load('../data/w2v/non_seq_model.model')
    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    print('Found %s word vectors.' % len(model.wv.vocab.items()))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embedding_matrix[i + 1] = vocab_list[i][1]
    print(embedding_matrix.shape)
    return embedding_matrix


embedding_matrix = embedding_w2v()
max_features = len(embedding_matrix)
print(embedding_matrix.shape)


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.001 for i in range(250,450)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


def plot_result(access_result):
    # 画loss图
    train_loss = np.array(access_result['train_loss'])
    x1 = train_loss[:, 0]
    y1 = train_loss[:, 1]
    print(x1)
    val_loss = np.array(access_result['val_loss'])
    x2 = val_loss[:, 0]
    y2 = val_loss[:, 1]
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.legend(['train', 'val'], loc='upper left')  # 图例 左上角
    plt.xlabel(u'epoch')
    plt.ylabel(u'loss')
    plt.title('Train History')
    plt.show()
    # 画accuracy图
    train_accuracy = np.array(access_result['train_accuracy'])
    x3 = train_accuracy[:, 0]
    y3 = train_accuracy[:, 1]
    val_accuracy = np.array(access_result['val_accuracy'])
    x4 = val_accuracy[:, 0]
    y4 = val_accuracy[:, 1]
    plt.plot(x3, y3)
    plt.plot(x4, y4)
    plt.legend(['train', 'val'], loc='upper left')  # 图例 左上角
    plt.xlabel(u'epoch')
    plt.ylabel(u'accuracy')
    plt.title('Train History')
    plt.show()
    # 画threshold_f1图
    val_threshold = np.array(access_result['val_threshold'])
    x5 = val_threshold[:, 0]
    y5 = val_threshold[:, 1]
    val_f1 = np.array(access_result['val_f1'])
    x6 = val_f1[:, 0]
    y6 = val_f1[:, 1]
    plt.plot(x5, y5)
    plt.plot(x6, y6)
    plt.legend(['threshold', 'f1'], loc='upper left')  # 图例 左上角
    plt.xlabel(u'epoch')
    plt.ylabel(u'threshold_f1')
    plt.title('Train History')
    plt.show()


# Lstm + Gru    Model2
def model_lstm_gru(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(GRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(1, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# CLR学习率策略
class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=300., mode='exp_range',
               gamma=0.99994)


def train_pred2(model, epochs=2):
    access_result = {}
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    val_threshold = []
    val_f1 = []
    total_time = 0
    for e in range(epochs):
        print('clr')
        print('epoch:',e)
        start_time = time.time()
        history = model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks=[clr])
        end_time = time.time()
        total_time = total_time + (end_time - start_time)
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        search_result = threshold_search(val_y, pred_val_y)
        print(search_result)
        # 保存threshold, f1
        threshold = [e, search_result['threshold']]
        val_threshold.append(threshold)
        f1 = [e, search_result['f1']]
        val_f1.append(f1)
        # 保存loss,accuracy
        t_loss = [e, history.history['loss'][0]]
        train_loss.append(t_loss)
        v_loss = [e, history.history['val_loss'][0]]
        val_loss.append(v_loss)
        t_acc = [e, history.history['acc'][0]]
        train_accuracy.append(t_acc)
        v_acc = [e, history.history['val_acc'][0]]
        val_accuracy.append(v_acc)
    pred_test_y = model.predict([X_test], batch_size=1024, verbose=0)
    pred_train_y = model.predict([train_X], batch_size=1024, verbose=0)

    print(val_threshold)
    access_result['train_loss'] = train_loss
    access_result['train_accuracy'] = train_accuracy
    access_result['val_loss'] = val_loss
    access_result['val_accuracy'] = val_accuracy
    access_result['val_threshold'] = val_threshold
    access_result['val_f1'] = val_f1
    avg_time = total_time / epochs
    print('average time:', avg_time)

    return pred_val_y, pred_test_y, pred_train_y, access_result


# best epochs = 37
# pred_val_y7, pred_test_y7, pred_train_y7, access_result= train_pred2(model_lstm_gru(embedding_matrix), epochs=41)
# print(pred_test_y7)
# y_true = Y_true
# y_predict = pred_test_y7 > 0.5
# print(y_predict)
# print('1. The F-1 score of the model {}\n'.format(f1_score(y_true, y_predict, average='macro')))
# print('2. The recall score of the model {}\n'.format(recall_score(y_true, y_predict, average='macro')))
# print('3. Classification report \n {} \n'.format(classification_report(y_true, y_predict)))
# print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true, y_predict)))
#
# y_true1 = train_y
# y_predict1 = pred_train_y7 > 0.5
# print('1. The F-1 score of the model {}\n'.format(f1_score(y_true1, y_predict1, average='macro')))
# print('2. The recall score of the model {}\n'.format(recall_score(y_true1, y_predict1, average='macro')))
# print('3. Classification report \n {} \n'.format(classification_report(y_true1, y_predict1)))
# print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_true1, y_predict1)))