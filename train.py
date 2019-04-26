import network
import tensorflow as tf
import numpy as np
import getData
import matplotlib.pyplot as plt

BATCH_SIZE = 32
LEARNING_RATE = 0.008
REGULARIZER = 0.01
KEEP_PROB = 0.5
EPOCH = 400

def train(train_set):
    x = tf.placeholder(tf.float32, shape=(None, network.IMAGE_RAW, network.IMAGE_COL, network.CHANNEL))
    y_ = tf.placeholder(tf.float32, shape=(None, network.OUTPUT_SIZE))
    y = network.forward(x, True, REGULARIZER, KEEP_PROB)
    loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)
    losses = tf.reduce_mean(loss)
    losses += tf.add_n(tf.get_collection('losses'))
    correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))  # argmax:1(按行算,得到列)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(losses)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        accu1=0
        accu2=0

        # 记录要绘图的变量
        x_p = [i for i in range(1, EPOCH + 1)]
        y_loss = [i for i in range(1, EPOCH + 1)]
        y_accu = [i for i in range(1, EPOCH + 1)]

        for i in range(1, EPOCH + 1):
            x_train, y_train = getData.get_batch(train_set, BATCH_SIZE)
            x_train = np.reshape(x_train, (BATCH_SIZE, network.IMAGE_RAW, network.IMAGE_COL, network.CHANNEL))

            _, loss_batch, accuracy_batch = sess.run([train, losses, accuracy],
                                                     feed_dict={x: x_train, y_: y_train})

            y_loss[i - 1] = loss_batch
            y_accu[i - 1] = accuracy_batch
            if i > 100 and accu1>0.4 and accu2>0.4 and accuracy_batch>0.4:
                saver.save(sess, 'ckpt_person/my_first.ckpt')

            accu1=accu2
            accu2=accuracy_batch

            print("epoch:", i)
            print("loss_batch:", loss_batch)
            print("accuracy_batch:", accuracy_batch)
            print(".....................................")

    print(y_loss)
    print(y_accu)
    plt.figure()
    plt.plot(x_p[0:len(x_p):4], y_loss[0:len(y_loss):4])
    plt.figure()
    plt.plot(x_p[0:len(x_p):4], y_accu[0:len(y_loss):4])
    plt.show()

train(getData.train_person)
