from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


class SVM(object):
    def __init__(self,lr = 0.01,ld = 0.01):
        self.lr = lr
        self.w = None
        self.b = None
        self.iter = None
        self.ld = ld
        pass

    def _init_params(self,ds):
        self.w = np.random.rand(len(ds[0]))
        self.b = np.random.rand()*10
        print(self.w,self.b)
        pass

    def _cal_grad(self,x,y):
        yx = 1-y*(self.w@x)+self.b
        if yx < 0:
            dw = self.ld*self.w
            db = 0
        else:
            dw = self.ld*self.w - y*x
            db = -y
            
        return dw, db

    def fit(self,ds,y,iter):
        self._init_params(ds)
        for i in range(iter):
            for j in range(len(y)):
                dw, db = self._cal_grad(ds[j],y[j])
                self.w -= dw*self.lr
                self.b -= db*self.lr
            print(dw, db)


    def predict(self,x,y):
        t = x @ self.w + self.b
        t[t<0] = -1
        t[t>0] = 1
        same = 0
        self.sarr = np.array([])
        for i in range(len(y)):
            if y[i] == t[i]:
                same += 1
                self.sarr = np.append(self.sarr,1)
            else:
                self.sarr = np.append(self.sarr,0)
        acc = same/200
        return acc

    def plot(self,ds,dy):
        ds = np.array(ds)
        x = np.arange(ds[:,0].min(),ds[:,0].max(),10)
        a = self.w
        m = -a[0]/a[1]
        c = self.b
        y = m*x+c
        plt.scatter(ds[:,0],ds[:,1], c = dy.reshape(200,1))
        plt.plot(x,y)
        plt.show()

    


# 2 blobs ; 100 each; 2d
ds, y = datasets.make_blobs(n_samples=200,n_features=2,centers=2,center_box=(0.0, 100.0))
y = (y-0.5)*2
model = SVM()
model.fit(ds,y,100)
print(model.predict(ds,y))
model.plot(ds,y)
