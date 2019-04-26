import tensorflow as tf
import network
import getData
import numpy as np


def test(data_set):
    x=tf.placeholder(dtype=tf.float32, shape=(None, network.IMAGE_RAW, network.IMAGE_COL, network.CHANNEL))
    y=network.forward(x, False, 0.1, 0.1)
    y_=tf.placeholder(dtype=tf.float32, shape=(None, network.OUTPUT_SIZE))

    losses=tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=y)
    loss=tf.reduce_mean(losses)
    correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)

        xs,ys=getData.get_all(data_set)
        xs=np.reshape(xs, (-1, network.IMAGE_RAW, network.IMAGE_COL, network.CHANNEL))

        model=tf.train.latest_checkpoint('ckpt_person/')
        saver.restore(sess,model)

        lo,accu=sess.run([loss,accuracy],feed_dict={x:xs,y_:ys})

        print("loss: ",lo)
        print("accuracy: ",accu)

test(getData.test_person)
