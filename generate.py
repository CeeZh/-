import numpy as np
import os
import random

# 数据：img(numpy)  图片大小： 120 X 128
# 标签：one-hot编码  direction(4)   emotion(4)   sunglasses(2)  person(20)

# 数据集：每个元素是一个二元组（数据，标签）
dataset_direction=[]
dataset_emotion=[]
dataset_sunglasses=[]
dataset_person=[]

# 训练集
train_direction=[]
train_emotion=[]
train_sunglasses=[]
train_person=[]

# 测试集
test_direction=[]
test_emotion=[]
test_sunglasses=[]
test_person=[]

# 训练集占比，剩余为测试集
rate=0.75


def judge_P2_P5(path):
    # 判断一个文件是P2还是P5
    # 返回：  P2  或  P5   或  WRONG
    with open(path,'rb') as f:
        line=f.readline().decode('utf-8')
        while True:
            line.strip()
            if line[0]=='#':
                line=f.readline()
                continue
            else:
                break
    if line=='P5\n':
        return 'P5'
    elif line=='P2\n':
        return 'P2'
    else:
        return 'WRONG'


def read_pgm_P2(path):
    # 读取路径为path的P2格式pgm图片
    # 输入：图片的路径path
    # 输出：ndarray格式的二维数组，图片大小（行，列），图片最大像素值（0为黑）

    with open(path) as f:
        lines=f.readlines()
    # 删除开始的空格
    for i in range(len(lines)):
        lines[i]=lines[i].strip()
    # 可能会有#作为图片描述，去除这些行
    newlines=[l for l in lines if l[0]!='#']
    # 去除每一行末尾的换行符
    newlines[1].strip('\n')
    for i in range(len(newlines)):
        newlines[i]= newlines[i].rstrip('\n')
    # 如果不是p2格式
    if newlines[0]!="P2":
        raise Exception("非P2格式")
    [col,raw]=[int(i) for i in newlines[1].split()]
    maxp=int(newlines[2])
    res=[]
    for l in newlines[3:]:
        res.extend([int(i) for i in l.split()])
    return np.array(res).reshape(raw,col),raw,col,maxp


def read_pgm_P5(path):
    # 读取路径为path的P2格式pgm图片
    # 输入：图片的路径path
    # 输出：numpy格式的二维数组，图片大小（行，列），图片最大像素值（0为黑）
    with open(path,'rb') as f:
        line=f.readline()
        if line.decode('utf-8')[0]=='#':
            l1=f.readline()
        if line.decode('utf-8')!='P5\n':
            raise Exception("非P5文件")
        line=f.readline()
        [col,raw]=[int(i) for i in line.decode('utf-8').strip('\n').split()]
        line=f.readline()
        maxp=int(line.decode('utf-8').strip('\n'))
        # 最大像素值小于256为一字节，否则2字节
        if maxp<256:
            one_reading=1
        else:
            one_reading=2
        img=np.zeros((raw,col))
        img[:, :] = [[ord(f.read(one_reading)) for j in range(col)] for i in range(raw)]
        return img,raw,col,maxp


def read_pgm(path):
    # 不需区分P2 P5 即可访问
    # 如果文件非P2P5，则返回False 否则返回img,raw,col,maxp
    t=judge_P2_P5(path)
    if t=='P2':
        return read_pgm_P2(path)
    elif t=='P5':
        return read_pgm_P5(path)
    else:
        raise Exception("非P2或P5文件")


def get_one_hot(labels,i):
    # 得到one_hot编码
    # 输入：labels   [person,direction,emotion,sunglasses]
    # 输入：i 正在操作的图片路径（便于察错）
    # 输出ndarray: person(20),direction(4),emotion(4),sunglasses(2)
    p=np.zeros((20))
    d=np.zeros((4))
    e=np.zeros((4))
    s=np.zeros((2))
    [person,direction,emotion,sunglasses]=labels

    if person=='an2i':
        p[0]=1
    elif person=='at33':
        p[1]=1
    elif person=='boland':
        p[2]=1
    elif person == 'bpm':
        p[3] = 1
    elif person=='ch4f':
        p[4]=1
    elif person=='cheyer':
        p[5]=1
    elif person=='choon':
        p[6]=1
    elif person=='danieln':
        p[7]=1
    elif person == 'glickman':
        p[8] = 1
    elif person=='karyadi':
        p[9]=1
    elif person=='kawamura':
        p[10]=1
    elif person=='kk49':
        p[11]=1
    elif person=='megak':
        p[12]=1
    elif person == 'mitchell':
        p[13] = 1
    elif person=='night':
        p[14]=1
    elif person=='phoebe':
        p[15]=1
    elif person == 'saavik':
        p[16] = 1
    elif person == 'steffi':
        p[17] = 1
    elif person == 'sz24':
        p[18] = 1
    elif person == 'tammo':
        p[19] = 1
    else:
        raise Exception("person not exist at "+i)

    if direction=="left":
        d[0]=1
    elif direction=='right':
        d[1]=1
    elif direction=='straight':
        d[2]=1
    elif direction=='up':
        d[3]=1
    else:
        raise Exception("direction not exist at "+i)

    if emotion=='angry':
        e[0]=1
    elif emotion=='happy':
        e[1]=1
    elif emotion=='neutral':
        e[2]=1
    elif emotion=='sad':
        e[3]=1
    else:
        raise Exception("emotion not exist at "+i)

    if sunglasses=='open':
        s[0]=1
    elif sunglasses=='sunglasses':
        s[1]=1
    else:
        raise Exception("sunglasses not exist at "+i)

    return p,d,e,s


def traverse(path):
    # 遍历path路径文件夹下的每个图片
    root=path
    list=os.listdir(path)
    for i in list:
        path=os.path.join(root,i)
        labels=i.split(".")[0].split("_")
        if(len(labels)==4):
            img, raw, col, maxp = read_pgm(path)
            # print(path)
            # print(raw,col,maxp)
            p,d,e,s=get_one_hot(labels,i)
            dataset_person.append([img,p])
            dataset_direction.append([img,d])
            dataset_emotion.append([img,e])
            dataset_sunglasses.append([img,s])


def group(data,rate):
    # 以rate的比例，把data分为train,test两部分
    # 输出：train,test

    #打乱
    random.shuffle(dataset_emotion)
    random.shuffle(dataset_sunglasses)
    random.shuffle(dataset_direction)
    random.shuffle(dataset_person)

    train=data[0:int(len(data)*rate)]
    test=data[int(len(data)*rate):]
    return train,test



root='./faces'
list=os.listdir(root)
for i in list:
    path=os.path.join(root,i)
    traverse(path)

train_direction, test_direction=group(dataset_direction, rate)
train_emotion, test_emotion=group(dataset_emotion, rate)
train_person, test_person=group(dataset_person, rate)
train_sunglasses, test_sunglasses=group(dataset_sunglasses, rate)

np.save("./train/direction",train_direction)
np.save("./train/emotion",train_emotion)
np.save("./train/person",train_person)
np.save("./train/sunglasses",train_sunglasses)

np.save("./test/direction", test_direction)
np.save("./test/emotion", test_emotion)
np.save("./test/person", test_person)
np.save("./test/sunglasses", test_sunglasses)








