# Imbalance data 
# https://datascienceschool.net/03%20machine%20learning/14.02%20%EB%B9%84%EB%8C%80%EC%B9%AD%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%AC%B8%EC%A0%9C.html
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


n0 = 200; n1 = 20

# Float data
rv1 = sp.stats.multivariate_normal([-1, 0], [[1, 0], [0, 1]])
rv2 = sp.stats.multivariate_normal([+1, 0], [[1, 0], [0, 1]])

#X0 = rv1.rvs(n0, random_state=0)
#X1 = rv2.rvs(n1, random_state=0)

# Int data
rng = np.random.default_rng()
X0 = rng.integers(100, size=(200,2))
X1 = rng.integers(100, size=(20,2))

X_imb = np.vstack([X0, X1])
y_imb = np.hstack([np.zeros(100), np.zeros(100)+3, np.ones(n1)])

x1min = -4; x1max = 4
x2min = -2; x2max = 2
xx1 = np.linspace(x1min, x1max, 1000)
xx2 = np.linspace(x2min, x2max, 1000)
X1, X2 = np.meshgrid(xx1, xx2)

def classification_result2(X, y, title=""):
    plt.contour(X1, X2, rv1.pdf(np.dstack([X1, X2])), levels=[0.05], linestyles="dashed")
    plt.contour(X1, X2, rv2.pdf(np.dstack([X1, X2])), levels=[0.05], linestyles="dashed")
    model = SVC(kernel="linear", C=1e4, random_state=0).fit(X, y)
    Y = np.reshape(model.predict(np.array([X1.ravel(), X2.ravel()]).T), X1.shape)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', label="0 클래스")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', label="1 클래스")
    plt.contour(X1, X2, Y, colors='k', levels=[0.5])
    y_pred = model.predict(X)
    plt.xlim(-4, 4)
    plt.ylim(-3, 3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    return model
X_imb = list(X_imb)
y_imb = list(y_imb)
X_samp, y_samp = SMOTE(random_state=4).fit_resample(X_imb, y_imb) #[220, 2] [220,]

print('done')

plt.subplot(121)
classification_result2(X_imb, y_imb)
plt.subplot(122)
model_samp = classification_result2(X_samp, y_samp)
plt.show()
