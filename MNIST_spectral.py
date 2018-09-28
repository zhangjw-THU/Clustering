from sklearn.cluster import KMeans
import numpy as np
import math as m
import matplotlib.pyplot as plt
import struct
import random
from sklearn.decomposition import PCA



def plot_digit(digit, label):
    """
    plot a num
    :param digit: 28*28
    :param label: label
    :return:
    """
    image = digit.reshape(28, 28)
    plt.imshow(image, cmap='gray_r', interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    plt.title("Label: " + str(label))
    plt.show()

def load_data( num,dataset="train"):
    """
    load data
    :param dataset:
    :return:
    """
    if dataset == "train":
        fname_img = 'train-images-idx3-ubyte'
        fname_lbl = 'train-labels-idx1-ubyte'
    elif dataset == "test":
        fname_img = 't10k-images-idx3-ubyte'
        fname_lbl = 't10k-labels-idx1-ubyte'
    else:
        raise ValueError("dataset must be either 'test' or 'train'")

    # load features
    fimg = open(fname_img, 'rb')
    magic, size = struct.unpack('>ii', fimg.read(8))
    if dataset == "train":
        size = num
    else:
        size = 10
    sx, sy = struct.unpack('>ii', fimg.read(8))
    img = []
    for i in range(size):
        im = struct.unpack('B' * (sx * sy), fimg.read(sx * sy))
        img.append([float(x) / 255.0 for x in im])

    fimg.close()
    X = np.array(img)

    # load labels
    flbl = open(fname_lbl, 'rb')
    magic, size = struct.unpack('>ii', flbl.read(8))
    if dataset == "train":
        size = num
    else:
        size = 10

    lbl = struct.unpack('B' * size, flbl.read(size))

    flbl.close()
    y = np.array(lbl)
    return X, y

class MySpectral:
    def __init__(self,k):
        self.K = k

    def get_dis_matrix(self,X):
        """
        X : tracks
        :return: distacne
        """

        nPoint = len(X)
        dis_matrix = np.zeros((nPoint, nPoint))

        for i in range(nPoint):
            for j in range(i + 1, nPoint):
                # dis_matrix[i][j] = dis_matrix[j][i] = m.sqrt(np.power(data[i] - data[j], 2).sum())
                dis_matrix[i][j] = dis_matrix[j][i] = np.linalg.norm(X[i] - X[j])

        return dis_matrix

    def getW(self,data, k):

        dis_matrix = self.get_dis_matrix(data)
        W = np.zeros((len(data), len(data)))
        for idx, each in enumerate(dis_matrix):
            index_array = np.argsort(each)
            W[idx][index_array[1:k + 1]] = 1
        tmp_W = np.transpose(W)
        W = (tmp_W + W) / 2
        return W

    def getD(self,W):

        D = np.diag(sum(W))
        return D

    def getL(self,D, W):
        return D - W

    def getEigen(self,L):
        eigval, eigvec = np.linalg.eig(L)
        ix = np.argsort(eigval)[0:self.K]
        return eigvec[:, ix]

    def plotRes(self,X, clusterResult):

        nPoints = len(X)
        scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
        for i in range(self.K):
            color = scatterColors[i % len(scatterColors)]
            x1 = []
            y1 = []
            for j in range(nPoints):
                if clusterResult[j] == i:
                    x1.append(X[j, 0])
                    y1.append(X[j, 1])
            plt.scatter(x1, y1, c=color, alpha=1, marker='*')
        plt.savefig("Spectral_2_" + str(random.randint(0, 100)) + ".jpg")
        plt.show()



if __name__ == '__main__':
    X_train, y_train = load_data(1000,"train")
    pca = PCA(n_components=2)
    X_train_pca = np.array(pca.fit_transform(X_train))

    k = 10
    KNN_K = 15

    sp = MySpectral(k)
    X_train_pca = np.asarray(X_train_pca)

    W = sp.getW(X_train_pca,KNN_K)
    D = sp.getD(W)
    L = sp.getL(D,W)
    eigvec = sp.getEigen(L)
    clf = KMeans(n_clusters=k)
    s = clf.fit(eigvec)
    c = s.labels_
    print(c)
    # nmi, acc, purity = eval.eva(C + 1, y_train)
    # print(nmi, acc, purity)
    sp.plotRes(X_train_pca, np.asarray(c))



