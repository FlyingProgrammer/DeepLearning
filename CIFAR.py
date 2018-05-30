import matplotlib.pyplot as plt
import pickle
import numpy as np
from PIL import Image
import cv2
lablesChinese=["飞机",'汽车','鸟','猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

#KNN算法
#输入：
#dataset：训练数据集 m*n
#lables： 每行数据集对应的标签
#input：  输入的数据集 1*n
#k：      比较范围
#输出：
#re       按相似性排序的结果
#dist     输入到所有训练集的距离
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
        for j in range(10,30): # j为测试数据的下标
            re,dist=KNN(x,y,xt[j],i)
            if(re[0][0]==yt[j]):
                right+=1
        k_v[0].append(i)
        k_v[1].append(right)
        if(right>cur):
            cur=right
            best_k=i
    drawplot(k_v)
    return best_k
#格式化单个数据
#将m*n*3按相同通道转为1*n
def formatArray(a):
    re=[]
    for i in range(32):
        for j in range(32):
            re.append(a[i][j][0])
    for i in range(32):
        for j in range(32):
            re.append(a[i][j][1])
    for i in range(32):
        for j in range(32):
            re.append(a[i][j][2])
    return re
#将按相同颜色通道合并的数据还原为m*n*3
def recover(all,max):
    result=[]
    all = all[:max, :]
    for a in all:
        a=np.array(list(a))
        a=np.reshape(a,(3,32*32))
        re=[]
        for i in range(32*32):
            cr=[a[0][i],a[1][i],a[2][i]]
            re.append(cr)

        re=np.reshape(re,(32,32,3))
        result.append(re)
    return np.array(result)
#根据文件名解压图片并保存，
##filename:cifar文件下的data_patch
def savetolocal(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    for i in range(len(dict.get(b"labels"))):
        m = dict.get(b"data")[i]
        l = dict.get(b"labels")[i]
        fn = dict.get(b"filenames")[i]
        m = thransform(m)
        image = Image.fromarray(m)
        #image.save("cifar\\" + str(l) + "\\" + fn.decode('utf-8'))
        image.save("cifar\\test" +  "\\" + str(i)+".png")
#
def load_cifar_traindata():
    with open("cifar\\data_batch_1", 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    dataset = dict.get(b"data")
    lables = dict.get(b"labels")
    return dataset,lables
#加载测试数据集
def load_cifar_testdata():
    with open("cifar\\test_batch",'rb') as f:
        dict=pickle.load(f,encoding='bytes')
    return dict.get( b'data'),dict.get( b'labels')
#测试本地图片
def recognizelocalimg(fp):
    #input=Image.open('cifar\\test\\'+str(i)+'.png')
    input = Image.open(fp)
    input=np.array(input)
    input=formatArray(input)
    print(lablesChinese[classifyone(input,3)])
#画散点图
def drawScatter(xdata,ydata,lables):
    dic={}
    for i in range(xdata.size):
        dic[lables[i]]=dic.get(lables[i],[])
        dic[lables[i]].append(xdata[i])
    dic2={}
    for i in range(ydata.size):
        dic2[lables[i]]=dic.get(lables[i],[])
        dic2[lables[i]].append(ydata[i])
    cvalue = ['darkblue', 'darkgreen', 'darkred', 'darkorange', 'dimgray', 'goldenrod', 'lightpink', 'black', 'red',
              'yellow']
    for i in range(10):
        datax=dic[i]
        datay=dic2[i]
        plt.scatter(datax, datay, s=5, c=cvalue[i])
    plt.show()
def demo():
    #demo 测试图片分类
    x,y=load_cifar_traindata()
    tx,ty=load_cifar_testdata()
    #K=3时有40%的准确率
    re,dist=KNN(x,y,tx[2],20,3)
    print(re,ty[2])
    #正确的调用方法
    showpictureorder(recover(tx,100),ty,(32,32,3))
x,y=load_cifar_traindata()
tx,ty=load_cifar_testdata()
testbest_K(x,y,tx,ty)