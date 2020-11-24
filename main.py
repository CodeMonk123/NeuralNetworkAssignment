import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from nn_regressor import NNRegressor


def f(x,y):
    return np.sin(x) - np.cos(y)

X_train1 = np.linspace(start=-6, stop=6, num=1024)
X_train2 = np.linspace(start=-6, stop=6, num=1024)
X_test1 = np.linspace(start=-5, stop=5, num=1000)
X_test2 = np.linspace(start=-5, stop=5, num=1000)
X_train = np.dstack(np.meshgrid(X_train1, X_train2)).reshape(-1, 2)
X_test = np.dstack(np.meshgrid(X_test1, X_test2)).reshape(-1, 2)

y_train= np.array(f(X_train[:,0],X_train[:,1]))
y_test = np.array(f(X_test[:,0],X_test[:,1]))

X_train = np.array(X_train,dtype=np.float64)
y_train = np.array(y_train,dtype=np.float64)
X_test = np.array(X_test,dtype=np.float64)
y_test = np.array(y_test,dtype=np.float64)

def plot_surf(X:np.array, Y:np.array, Z:np.array, name:str):
    fig = plt.figure(figsize=(8,6), dpi=150)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues,
                       linewidth=0, antialiased=False)
    ax.set_zlim(-3, 3)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(name)

def plot_loss(training_loss:[float]):
    fig = plt.figure(figsize=(8,6), dpi=150)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.plot(training_loss)
    plt.savefig('./loss.png')
    

if __name__ == "__main__":
    regressor = NNRegressor(max_epochs=1000)
    training_loss = regressor.fit(X=X_train, y=y_train)
    y_predict = regressor.predict(X_test)
    y_predict = np.reshape(y_predict, (1000,1000))
    predict_error = np.abs(np.reshape(y_test,(1000,1000)) - y_predict)**2

    X,Y = np.meshgrid(X_test1,X_test2)
    plot_surf(X,Y, y_predict, './predict.png')
    print(np.shape(predict_error))
    plot_surf(X,Y, predict_error, './error.png')
    plot_surf(X,Y, np.reshape(y_test,(1000,1000)), './real.png')
    plot_loss(training_loss)
