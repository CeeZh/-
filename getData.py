import numpy as np
import random
import math
from matplotlib import pyplot as plt


# 训练集
train_direction=np.load("./train/direction.npy")
train_emotion=np.load("./train/emotion.npy")
train_person=np.load("./train/person.npy")
train_sunglasses=np.load("./train/sunglasses.npy")


# 测试集
test_direction=np.load("./test/direction.npy")
test_emotion=np.load("./test/emotion.npy")
test_person=np.load("./test/person.npy")
test_sunglasses=np.load("./train/sunglasses.npy")


def get_batch(data,batch):
    # 在data数据集里随机选batch个数据
    # 返回值：数据（原版ndarray），标签(onehot)
    xs=[]
    ys=[]
    length = len(data)
    position=[i for i in range(0,length)]
    pos=random.sample(position,batch)
    for i in pos:
        xs.append(data[i][0])
        ys.append(data[i][1])
    return xs,ys


def get_train_validation(data,k,no):
    # 把train均分k份，以第no份当validation，no从0开始计数
    # 返回train test
    length=len(data)
    print(length)
    d=math.ceil(length/k)
    start=d*no
    end=start+d
    if end>length:
        end=d-length+start
        return data[end:start],np.concatenate([data[start:],data[0:end]],0)
    else:
        return np.concatenate([data[0:start],data[end:]],0),data[start:end]


def get_all(data):
    # 用于得到测试集（全部）
    xs=[]
    ys=[]
    for i in range(len(data)):
        xs.append(data[i][0])
        ys.append(data[i][1])
    return xs,ys

