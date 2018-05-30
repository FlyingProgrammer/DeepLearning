import numpy as np
import matplotlib.pyplot as plt
import struct
import cv2
#手写数字数据结构
#训练数据集trainset：前16个字符表示文件的基本信息 60000*784
#训练集赌赢的标签结果trainlabels：前8个字符表示文件的基本信息 60000*1
def load_mnist_traindata():
    with open(r"data\train-labels.idx1-ubyte",'rb') as f:
        magic, n = struct.unpack('>II',f.read(8))#前8个字符表示文件的基本信息
        trainlabels = np.fromfile(f, dtype=np.uint8)
    with open(r"data\train-images.idx3-ubyte",'rb') as f:
        trainset= struct.unpack('>IIII',f.read(16))   #前16个字符表示文件的基本信息
        trainset  = np.reshape(np.fromfile(f,dtype=np.uint8),(len(trainlabels),28*28))
    return trainset,trainlabels
def load_mnist_testdata():
    with open(r"data\t10k-labels.idx1-ubyte",'rb') as f:
        magic, n = struct.unpack('>II',f.read(8))
        trainlabels = np.fromfile(f, dtype=np.uint8)
    with open(r"data\t10k-images.idx3-ubyte",'rb') as f:
        trainset= struct.unpack('>IIII',f.read(16))
        trainset  = np.reshape(np.fromfile(f,dtype=np.uint8),(len(trainlabels),28*28))

    return trainset,trainlabels
#KNN算法
def KNN(dataset,lables,input,k=4):
    input=np.tile(input,[dataset.shape[0],1])
    dist = np.sqrt(np.sum(np.square(dataset - input), axis=1))
    indexs=np.argsort(dist)
    re={}
    for i in range(k):
        lable=lables[indexs[i]]
        re[lable]=re.get(lable,0)+1
    re=sorted(re.items(), key=lambda d: d[1],reverse=True)
    return re,dist

#按顺序显示图片 x_train:m*784
def showpictureorder(x_train,y_train,shape=(28,28),rows=5,cols=10):
    y_train=np.array(y_train)
    x_train=np.array(x_train)
    ig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True, )
    ax = ax.flatten()

    for i in range(rows):
        for j in range(cols):
            img = x_train[i*cols+j].reshape(shape)
            ax[i*cols+j].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
#显示一种种类的图片
def showonekind(x_train,y_train,label_index,shape=(28,28),rows=5,cols=10):
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    ig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,  # 显示坐标轴
        sharey=True, )
    ax=ax.flatten()
    for i in range(rows):
        for j in range(cols):
            img = x_train[y_train==label_index][i*cols+j].reshape(shape)
            ax[i*cols+j].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()   #调整图像边缘及图像间的空白间隔
    plt.show()
#预处理 简单的二值化
def proprocess(xdata):
    xdata = cv2.threshold(xdata, 127, 1, cv2.THRESH_BINARY)
    return xdata

#按种类显示图片
def showpicturekind(x_train,y_train,shape=(28,28),rows=5,cols=10):
    y_train=np.array(y_train)
    x_train=np.array(x_train)
    ig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True, #显示坐标轴
        sharey=True, )
    ax = ax.flatten() #折叠成一维的数组

    for i in range(rows):
        for j in range(cols):
            img = x_train[y_train == j][i].reshape(shape)
            ax[i*cols+j].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()   #调整图像边缘及图像间的空白间隔
    plt.show()

def drawplot(data):
    plt.figure(figsize=(10, 8), dpi=100)
    plt.plot(data[0], data[1], 'r',label="K")
    plt.show()
def testbest_K(x,y,xt,yt):
    best_k=0
    cur=0
    k_v=[[],[]]
    for i in range(1,10):#i为k的值
        right = 0
        for j in range(1200,1220): # j为测试数据的下标
            re,dist=KNN(x,y,xt[j],i)
            if(re[0][0]==yt[j]):
                right+=1
        k_v[0].append(i)
        k_v[1].append(right)
        print(i,right)
        if(right>cur):
            cur=right
            best_k=i
    drawplot(k_v)
    return best_k
def demo():
    x,y=load_mnist_traindata()
    tx,ty=load_mnist_testdata()
    re,dist=KNN(x,y,tx[0])
    print(re,ty[0])
demo()