import pickle
from random import randint
import numpy as np
import tensorflow as tf
import multiprocessing as mp

# GLOBAL
x = pickle.load(open("essays_mairesse.p", "rb"))
# loading form process_data.py
# W2 won't be used
revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
# number of channels
ch = 200
# max number of sentences in an essay
n_s = 0
for i in range(len(revs)):
    temp = len(revs[i]['text'])
    n_s = max(n_s, temp)

# max length of sentence in an essay
def max_sentence(essay):
    l_s = [len(s.split(' ')) for s in essay]
    return max(l_s)

# number of essays
n_essay = len(revs)
# list of longest sentences in each essay
l_long = [max_sentence(essay['text']) for essay in revs]
# maximum number of words in a sentence
w = max(l_long)

# convert sentence to a vector with length w*300
def s_to_vec(s, w):
    # convert words in s to list vectors
    l_vec = [W[word_idx_map[i],] for i in s.split(' ')]
    # length of sentence
    k = len(l_vec)
    v = np.asarray(l_vec)
    # padding
    for i in range(k, w):
        v = np.append(v, np.zeros([1, 300]), axis=0)
    v = v.reshape(w * 300)
    return v


# convert essay to an array
def e_to_l(e):
    k = len(e)
    l = []
    for s in e:
        l.append(s_to_vec(s, w))
    # do padding for essays with only one sentence
    if k == 1:
        l.append(np.zeros(w*300))
    array = np.asarray(l)
    return array


# convert label to vector
def y_to_v(y):
    if y == 1:
        a = [-1.0, 1.0]
    else:
        a = [1.0, -1.0]
    vec = np.asarray(a).reshape([2, 1])
    return vec


# create feed dict
# l_ind indices of training data in revs
def feed_dict(l_ind):
    l, j = len(l_ind), 0
    print('Starting creating dict')
    # m_train: list of Mairesse features
    x_train, y_train, m_train = [], [], []
    for i in l_ind:
        labels = [y_to_v(revs[i]['y0']), y_to_v(revs[i]['y1']), y_to_v(revs[i]['y2']), y_to_v(revs[i]['y3']),
                  y_to_v(revs[i]['y4'])]
        labels = np.asarray(labels)
        y_train.append(labels)
        m_train.append(mairesse[revs[i]['user']])
        x_train.append(e_to_l(revs[i]['text']))
        l -= 1
        print(l, ' remaining')
    x_train, y_train, m_train = np.asarray(x_train), np.asarray(y_train), np.asarray(m_train)
    print('Done')
    return x_train, y_train, m_train


# input a tensor with shape 1 * 444000, n-grams
def nn_1(ts, n):
    mat_s = tf.reshape(ts, [1, w, 300, 1])
    # first layer with relu
    conv1 = tf.layers.conv2d(mat_s, filters=ch, kernel_size=(n, 300), activation=tf.nn.leaky_relu)
    return conv1


def main():
    # indices of training data
    np.random.seed(1)
    l_id = list(range(n_essay))
    np.random.shuffle(l_id)
    train_id, test_id = l_id[:int(.8 * n_essay)], l_id[int(.8 * n_essay):]
    # splitting
    x_train, y_train, m_train = feed_dict(train_id)
    x_test, y_test, m_test = feed_dict(test_id)
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    #pickle.dump([train_id, test_id, x_train, y_train, x_test, y_test], open("train_test.p", "wb"))

    # output file names of different personality traits
    traits = ['EXT.p', 'NEU.p', 'AGR.p', 'CON.p', 'OPN.p']
    # n-gram and k th label
    # k in range(5)
    for k_label in range(5):

        epoch = 100
        nn_pred, uni, bi, tri = {}, {}, {}, {}
        ubt = [nn_pred, uni, bi, tri]

        for n in [1,2,3]:
            print('creating features of', n,'gram')
            # NN with SGD, feed an essay at a time
            essay = tf.placeholder(dtype=tf.float32, shape=[None, w * 300])
            # with shape [n_sen, w - n+1, ch]
            conv_1 = tf.squeeze(tf.map_fn(lambda s: nn_1(s, n), essay))
            # max pooling  on the dimension of w-n+1, out_2 with shape [n_sen, ch]
            pool_1 = tf.math.reduce_max(conv_1, axis=1)
            # 1d tensor with length ch
            pool_1 = tf.math.reduce_max(pool_1, axis=0)
            # dropout
            pool_1 = tf.layers.dropout(pool_1, 0.6)
            # shape [ch, 1]
            pool_1 = tf.expand_dims(pool_1, 1)

            # linear layer
            wt_2 = tf.Variable(tf.random_uniform([2, ch]))
            b_2 = tf.Variable(tf.random_uniform([2, 1]))
            y = tf.nn.sigmoid(tf.matmul(wt_2, pool_1) + b_2)

            # conv_2 = tf.matmul(wt_2, pool_1)
            # prediction
            # y = tf.nn.softmax(conv_2, 0)
            # label e.g.[0,1]
            y_true = tf.placeholder(dtype=tf.float32, shape=[2, 1])

            # cross entropy
            # loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), 0))
            loss = -tf.reduce_mean(y_true * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_true) * tf.log(
                tf.clip_by_value(1 - y, 1e-10, 1.0)))
            optimizer = tf.train.GradientDescentOptimizer(0.5)
            train_step = optimizer.minimize(loss)

            init = tf.global_variables_initializer()

            #sess = tf.Session()
            #sess.run(init)
            #print(sess.run(essay, {essay: x_train[10], y_true: y_train[10][0]}))
            #a = sess.run(pool_1, {essay: x_train[2], y_true: y_train[2][0]})
            #sess.close()

            sess = tf.Session()
            sess.run(init)
            # train step
            for j in range(epoch):
                print(j)
                i = randint(0, n_train-1)
                sess.run(train_step, {essay: x_train[i], y_true: y_train[i][k_label]})
                #print(sess.run(loss, {essay: x_train[i], y_true: y_train[i][k_label]}))

            # extract 1*200 vec of n-gram
            for j in range(n_train):
                print(j, 'is done in train set')
                ind = train_id[j]
                ID =revs[ind]['user']
                feature = sess.run(pool_1, {essay: x_train[j], y_true: y_train[j][k_label]})
                pred = sess.run(tf.arg_max(y, 0), {essay: x_train[j], y_true: y_train[j][k_label]})
                ubt[n][ID] = feature.flatten().tolist()
                nn_pred[ID] = int(pred)

            for j in range(n_test):
                print(j, 'is done in test set')
                ind = test_id[j]
                ID =revs[ind]['user']
                feature = sess.run(pool_1, {essay: x_test[j], y_true: y_test[j][k_label]})
                pred = sess.run(tf.arg_max(y, 0), {essay: x_test[j], y_true: y_test[j][k_label]})
                ubt[n][ID] = feature.flatten().tolist()
                nn_pred[ID] = int(pred)
            sess.close()
            print(n,'gram is all set')
        ####################################
        # save data for classification
        pickle.dump([revs, vocab, mairesse, uni, bi, tri, train_id, test_id, nn_pred], open(traits[k_label], "wb"))
# load data
# temp = pickle.load(open("data_for_class.p", "rb"))
# revs, train_id, test_id are lists, others are all dicts
# values of uni, bi, tri are vector with length 200
# nn_pred is dict of predictions made by cnn
# revs, vocab, mairesse, uni, bi, tri, train_id, test_id, nn_pred = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
####################################
if __name__=='__main__':
    main()

