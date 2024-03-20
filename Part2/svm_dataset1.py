import pickle
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def model_display_boundary(X, model, label):
    h = 0.01  # step size in the mesh, we can decrease this value for smooth plots, i.e 0.01 (but ploting may slow down)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    aa = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(aa)
    print(Z, set(Z))
    Z = Z.reshape(xx.shape)
    # plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.contourf(xx, yy, Z, c=Z, alpha=0.25)  # cmap="Paired_r",
    # plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=label, cmap="Paired_r", edgecolors="k")
    x_ = np.array([x_min, x_max])

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))
cs = [0.1, 1]
kernels = ["rbf", "sigmoid", "poly"]


for c in cs:
    for kernel in kernels:
        if kernel == "poly":
            svm = SVC(C=c, kernel=kernel, degree=10)
        else:
            svm = SVC(C=c, kernel=kernel)  # type: ignore
        svm.fit(dataset, labels)
        predicted = svm.predict(dataset)
        print(
            "C: ",
            c,
            " Kernel: ",
            kernel,
            " Accuracy: ",
            accuracy_score(labels, predicted),
            " F1 Score: ",
            f1_score(labels, predicted),
        )
        model_display_boundary(dataset, svm, labels)
