import numpy as np
import matplotlib.pyplot as plt
import struct
import sklearn
from sklearn.decomposition import PCA
import random



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


def load_data(num,dataset="train"):
    """
    data load
    :param dataset: train or test
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
        size = num
    else:
        size = 500

    lbl = struct.unpack('B' * size, flbl.read(size))

    flbl.close()
    y = np.array(lbl)

    return X, y


class MyKMeans:

        def __init__(self, k):
            """
            initial
            :param k: clustring num
            """
            self.K = k
            self.centers = None

        def initialize_centers_random(self, X):
            """
            initialize centers with randomly chosen points
            :param X: data
            :return:
            """
            k = self.K
            centers = []
            random.seed(2018)
            for _ in range(k):
                centers.append(X[random.randint(0, len(X))])
            self.centers = centers

        def initialize_centers_specific(self, X):
            """
            initialize centers with "0,1,2,...."
            :param X: data
            :return:
            """
            k = self.K
            if k != 10:
                print("if you chose initial specific , k must equal 10!")
                return 0

            centers = [X[1], X[3], X[5], X[7], X[2], X[0], X[13], X[15], X[17], X[4]]  # 0~9

            self.centers = centers

        def assign_points(self, X):
            """
            assigns data point to their nearst centers
            :param X: data point
            :return:
            """
            assignment = []
            dist = []

            for x in X:
                temp = []
                for center in self.centers:
                    temp.append(np.linalg.norm(center - x))
                dist.append(temp)
            for d in dist:
                assignment.append(d.index(min(d)))

            return assignment

        def update_centers(self, X, assignment):
            """
            updata centers with num means
            :param X: data points
            :param assignment: clustring
            :return: new points
            """

            clusters = [[] for _ in range(self.K)]
            for i, x in zip(assignment, X):
                clusters[i].append(x)

            centroids = [np.mean(cluster, axis=0) for cluster in clusters]
            self.centers = centroids

            return clusters

        def compute_WCSS(self, X):
            """
            computer within cluster sum of squres
            :param X:
            :return:
            """

            wcss = 0
            for i, centroid in zip(range(len(self.centers)), self.centers):
                for x in X[i]:
                    wcss = wcss + np.linalg.norm(x - centroid)

            return wcss


        def fit(self, X, max_iter=20):
            """
            train KMeans
            :param X: data
            :param max_iter: epoch
            :return: centers and clustring
            """

            errors = []
            wcss = 0.
            for i in range(max_iter):
                print ('Epoch : ', i)

                assignments = km.assign_points(X)
                clusters = km.update_centers(X, assignments)
                if abs(wcss - km.compute_WCSS( clusters )) <= 0.1:
                    print('Optimal found in ', i, 'epoch')
                    break
                wcss = km.compute_WCSS(clusters)
                errors.append(wcss)
                print ('WCSS = ', wcss)
            return clusters , assignments , errors

        def plotRes(self, X, assigments):
            """
            show of answer
            :param X:
            :return:
            """
            scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
            for i in range(len(assigments)):
                color = scatterColors[assigments[i] % len(scatterColors)]
                x1 = [X[i][0]]
                y1 = [X[i][1]]
                plt.scatter(x1, y1, c=color, alpha=1, marker='*')
            plt.savefig("KMeans_specific" + str(random.randint(0, 100)) + ".jpg")
            plt.show()


if __name__ == '__main__':
    X_train, y_train = load_data(500,"train")
    # X_test, y_test = load_data("test")
    # plot_digit(X_train[0])
    n = 2
    pca = PCA( n_components=n )
    X_train_pca = np.array(pca.fit_transform(X_train))
    # a_s = pca.inverse_transform(X_train_pca)
    k = 10
    km = MyKMeans(k)
    # km.initialize_centers_random(X_train_pca)
    km.initialize_centers_specific(X_train_pca)

    clusters,assigments,error = km.fit(X_train_pca)

    print("Kmeans (k=10) of MNIST :")
    cluster_result = [[] for _ in range(k)]
    for i in range(len(X_train_pca)):
        cluster_result[assigments[i]].append(y_train[i])
    for i in range(k):
        print("clustring ",i," including points: ",len(cluster_result[i]))
        statistics = []
        for j in range(10):
            num = 0
            for k in range(len(cluster_result[i])):
                if cluster_result[i][k] == j:
                    num +=1
            statistics.append(num)

        print("the num of each digital (0~9): ",statistics)
        print(statistics.index(max(statistics)),"account :",float(max(statistics))/float(len(cluster_result[i])+0.1))
    if n == 2:
        km.plotRes(X_train_pca,assigments)
