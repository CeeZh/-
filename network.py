import tensorflow as tf

IMAGE_RAW=120
IMAGE_COL=128
CHANNEL=1

CONV1_SIZE=3
CONV1_NUM=64

POOL1_SIZE=2
POOL1_STRIDE=2

CONV2_SIZE=3
CONV2_NUM=64

POOL2_SIZE=2
POOL2_STRIDE=2

CONV3_SIZE=3
CONV3_NUM=64

POOL3_SIZE=2
POOL3_STRIDE=2

# CONV4_SIZE=3
# CONV4_NUM=64
#
# POOL4_SIZE=2
# POOL4_STRIDE=2

FC1_SIZE=256
FC2_SIZE=128
OUTPUT_SIZE=20

# 用于添加正则化项
def get_weight(shape, regularizer, train):
    # 只有训练集上用正则化，train=true
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if train:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def forward(x, train, regularizer, keep_prob):
    # 输入x: [batch_size,raw,col,channel]
    # train: 训练集为true，测试集为false。  Dropout开关
    # regulizer: 正则化系数
    # keep_prob: 用于droput

    # 返回output: [batch_size, OUTPUT_SIZE]

    # CONV1 : [batch_size,raw,col,CONV1_SIZE]
    # 卷积核描述：[行，列，通道数，个数]
    conv1_w=get_weight((CONV1_SIZE,CONV1_SIZE,CHANNEL,CONV1_NUM), regularizer, train)
    conv1_b=tf.Variable(tf.zeros(shape=(CONV1_NUM)))
    conv1=tf.nn.conv2d(input=x,filter=conv1_w,strides=[1,1,1,1],padding='SAME')
    conv1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))

    # POOL1 : [b,raw/POOL1_SIZE,col/POOL1_SIZE,CONV1_SIZE]
    # 池化ksize: [1,行，列，1]     strides: [1,行，列，1]
    pool1=tf.nn.max_pool(conv1,ksize=[1,POOL1_SIZE,POOL1_SIZE,1],strides=[1,POOL1_STRIDE,POOL1_STRIDE,1],padding='SAME')

    # CONV2 : [b,raw/POOL1_SIZE,col/POOL1_SIZE,CONV2_SIZE]
    conv2_w=get_weight((CONV2_SIZE,CONV2_SIZE,CONV1_NUM,CONV2_NUM), regularizer, train)
    conv2_b=tf.Variable(tf.zeros(shape=(CONV2_NUM)))
    conv2=tf.nn.conv2d(pool1,conv2_w,[1,1,1,1],'SAME')
    conv2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))

    # POOL2 : [b,raw/POOL1_SIZE/POOL2_SIZE,col/POOL1_SIZE/POOL2_SIZE,CONV2_SIZE]
    pool2=tf.nn.max_pool(conv2,[1,POOL2_SIZE,POOL2_SIZE,1],[1,POOL2_STRIDE,POOL2_STRIDE,1],'SAME')

    conv3_w=get_weight((CONV3_SIZE,CONV2_SIZE,CONV2_NUM,CONV3_NUM),regularizer,train)
    conv3_b=tf.Variable(tf.zeros((CONV3_NUM)))
    conv3=tf.nn.conv2d(pool2,conv3_w,[1,1,1,1],'SAME')
    conv3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_b))

    pool3=tf.nn.max_pool(conv3,[1,POOL3_SIZE,POOL3_SIZE,1],[1,POOL3_STRIDE,POOL3_STRIDE,1],'SAME')

    # conv4_w = get_weight((CONV4_SIZE, CONV3_SIZE, CONV3_NUM, CONV4_NUM), regularizer, train)
    # conv4_b = tf.Variable(tf.zeros((CONV4_NUM)))
    # conv4 = tf.nn.conv2d(pool3, conv4_w, [1, 1, 1, 1], 'SAME')
    # conv4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_b))
    #
    # pool4 = tf.nn.max_pool(conv4, [1, POOL4_SIZE, POOL4_SIZE, 1], [1, POOL4_STRIDE, POOL4_STRIDE, 1], 'SAME')

    # FC0(展平)
    # FC0 : [batch_size,n]
    pool4_shape=pool3.get_shape().as_list()
    fc0_size=pool4_shape[1]*pool4_shape[2]*pool4_shape[3]
    fc0=tf.reshape(pool3,shape=(-1,fc0_size))

    # FC1 : [batch_size,FC1_SIZE]
    fc1_w=get_weight((fc0_size,FC1_SIZE), regularizer, train)
    fc1_b=tf.Variable(tf.zeros(shape=(FC1_SIZE)))
    fc1=tf.matmul(fc0,fc1_w)+fc1_b     # 广播机制
    fc1=tf.nn.relu(fc1)
    if train:
        fc1=tf.nn.dropout(fc1,keep_prob)

    # FC2
    fc2_w=get_weight((FC1_SIZE,FC2_SIZE), regularizer, train)
    fc2_b=tf.Variable(tf.zeros(shape=(FC2_SIZE)))
    fc2=tf.matmul(fc1,fc2_w)+fc2_b
    fc2=tf.nn.relu(fc2)
    if train:
        fc2=tf.nn.dropout(fc2,keep_prob)

    # OUTPUT (不用激活函数)
    fc3_w=get_weight((FC2_SIZE,OUTPUT_SIZE),regularizer,train)
    fc3_b=tf.Variable(tf.zeros(shape=(OUTPUT_SIZE)))
    output=tf.matmul(fc2,fc3_w)+fc3_b

    return output



