import numpy as np
import matplotlib.pyplot as plt
import struct
import random
import sklearn
from sklearn.decomposition import PCA




def plot_digit(digit, label=0):
    """
    show
    :param digit: array
    :param label:
    :return:
    """

    image = digit.reshape(28, 28)

    plt.imshow(image, cmap='gray_r', interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    plt.title("Label: " + str(label))
    plt.show()


def load_data(dataset="train"):
    """
    data load
    :param dataset: train or test
    :return:
    """
    num_train_point = 20
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
        size = num_train_point
    else:
        size = 500
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
        size = num_train_point
    else:
        size = 500

    lbl = struct.unpack('B' * size, flbl.read(size))

    flbl.close()
    y = np.array(lbl)

    return X, y

class MyHierarchical:
    def __init__(self,k):
        """
        :param k: num of clustring
        """
        self.K = k


    def dist(self, a, b):
        """
        the dist between two points
        :param a: point a
        :param b: point b
        :return:
        """
        d = np.linalg.norm(a - b)
        return d

    def dist_min(self, Ci, Cj):
        """
        get shortest distance
        :param Ci: Clustring A
        :param Cj: Clustring B
        :return:
        """
        return min(self.dist(a, b) for a in Ci for b in Cj)

    def dist_max(self, Ci, Cj):
        """
        get the farthest distance
        :param Ci:
        :param Cj:
        :return:
        """
        return max(self.dist(a, b) for a in Ci for b in Cj)

    def dist_avg(self, Ci, Cj):
        """
        get average distance
        :param Ci:
        :param Cj:
        :return:
        """
        return sum(self.dist(a, b) for a in Ci for b in Cj) / (len(Ci) * len(Cj))

    def find_min(self, M):
        """
        find the nearest clustering
        :param M: all clustering
        :return:
        """
        min_d = M[0][1]
        x = 0;
        y = 0
        for i in range(len(M)):
            for j in range(len(M[i])):
                if i != j and M[i][j] < min_d:
                    min_d = M[i][j]
                    x = i
                    y = j
        return (x, y, min_d)

    def fit(self, x_train, y_train, dist):
        """
        train the data
        :param x_train: points
        :param y_train: labels
        :param dist: dist function
        :return:
        """
        k = self.K
        # dist = self.dist_max
        C = []
        M = []
        C_y = []
        for i in x_train:
            Ci = []
            Ci.append(i)
            C.append(Ci)
        for i in y_train:
            Ci_y = []
            Ci_y.append(i)
            C_y.append(Ci_y)

        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)

        q = len(x_train)

        while q > k:
            x, y, min_d = self.find_min(M)

            C[x].extend(C[y])
            C_y[x].extend(C_y[y])
            c_t = []
            c_t_y = []

            # print(000)
            for i in range(len(C)):
                if i == int(y):
                    continue
                c_t.append(C[i])
                c_t_y.append(C_y[i])
            C = c_t
            C_y = c_t_y
            M = []
            for i in C:
                Mi = []
                for j in C:
                    Mi.append(dist(i, j))
                M.append(Mi)
            q = q - 1
            print(q)
            # print(C_y)
            # print(C)
        return C , C_y, q

    def plotRes(self,X):
        """
        show of answer
        :param X:
        :return:
        """
        scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
        for j in range(len(X)):
            color = scatterColors[j % len(scatterColors)]
            for i in range(len(X[j])):
                x1 = [X[j][i][0]]
                y1 = [X[j][i][1]]
                plt.scatter(x1, y1, c=color, alpha=1, marker='*')
        plt.savefig("Hierarchical_"+str(random.randint(0,100))+".jpg")
        plt.show()



if __name__ == "__main__":
    X_train, y_train = load_data("train")
    # X_test, y_test = load_data("test")
    n = 2
    pca = PCA(n_components=n)
    X_train_pca = np.array(pca.fit_transform(X_train))
    # a_s = pca.inverse_transform(X_train_pca)
    k = 10

    hc = MyHierarchical(k)
    C, C_y, q = hc.fit(X_train_pca,y_train,hc.dist_avg)
    print("Hierarchical Clustring (k=10) of MNIST :")
    print(C_y)



    for i in range(k):
        print("clustring ", i, " including points: ", len(C_y[i]))
        statistics = []
        for j in range(10):
            num = 0
            for k in range(len(C_y[i])):
                if C_y[i][k] == j:
                    num += 1
            statistics.append(num)

        print("the num of each digital (0~9): ", statistics)
        print(statistics.index(max(statistics)), "account :", float(max(statistics)) / float(len(C_y[i])))
    if n == 2:
        hc.plotRes(C)



