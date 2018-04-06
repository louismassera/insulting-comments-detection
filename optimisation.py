import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.sparse import *
from scipy.sparse.linalg import *
import sys
sys.setrecursionlimit(10000)

class StochasticGradient:
    def __init__(self,loss='log',tol=0.00010,penalty='l2',C=1.0,alpha=0.0001,n_iter=10,learning_rate='optimal',classes_weight=None):
        self.tol_ = tol
        self.penalty_ = penalty
        self.C_ = C
        self.loss_ = loss
        self.alpha_ = alpha
        self.n_iter_ = n_iter
        self.classes_weight_ = classes_weight
        if (learning_rate == 'optimal'):
            self.learning_rate_ = learning_rate
        else:
            self.learning_rate_ = learning_rate

    def fit(self,X,y):
        ## descente
        X = X.toarray()
        B = np.array([[1] for i in range(X.shape[0])])
        X = np.hstack([X,B])
        if (np.unique(y) == np.array([0,1])).all():
            y = 2*y-1

        self._get_weight(X,y,cas=self.classes_weight_)

        w = 0.3 * np.ones((X.shape[1],))
        self.n_samples = X.shape[0]
        self.p_features = X.shape[1]
        eta =1
        imax = self.n_iter_
        w,step,delta = self._grad_descent(X,y,w,eta,imax)
        self.step_ = step
        self.delta_ = delta
        self.w_ = w

        return self
    def fit2(self,X,y):
        X = X.toarray()
        B = np.array([[1] for i in range(X.shape[0])])
        X = np.hstack([X,B])
        if (np.unique(y) == np.array([0,1])).all():
            y = 2*y-1

        self._get_weight(X,y,cas=self.classes_weight_)

        w = np.zeros((X.shape[1],))
        self.n_samples = X.shape[0]
        self.p_features = X.shape[1]
        w,step,delta = self._grad_descent2(X,y,w)
        self.step_ = step
        self.delta_ = delta
        self.w_ = w

    def predict(self,X):
        ## prediction
        X = X.toarray()
        B = np.array([[1] for i in range(X.shape[0])])
        X = np.hstack([X,B])
        y_pred = np.sign(X.dot((self.w_).T))
        return y_pred

    def score(self,X,y):
        ## compute score
        return np.mean(self.predict(X) == y)

    def _get_weight(self,X,y,cas):
        ## get classes weight
        if(cas in ['balanced','unbalanced']):
            self.weight_ = float(len(y)) / (2*np.bincount(((y+1.0)/2).astype(int)))
            if(cas == 'unbalanced'):
                self.weight_ = np.array([self.weight_[1],self.weight_[0]])
        elif(cas == None):
            self.weight_ = np.ones((2,))
        return self


    def _grad_descent2(self,x,y,w):
        learning_rate = self.learning_rate_
        t0 = 1
        thresh = self.tol_
        n_iter = self.n_iter_
        alpha = self.alpha_
        n_iter_max = self.n_samples
        p = self.p_features -1
        u = w.copy()
        gamma = 0.1
        a = 0.5
        beta = 0.5
        step = 0
         ## begin storage
        while(np.linalg.norm(gradf2(u,x,y,p,n_iter_max))>thresh):
            b = 2*gamma
            gamma = b
            u_ = proximal_g(u - gamma * gradf2(u,x,y,p,n_iter_max),gamma,p,alpha)
            while(f2(u_,x,y,n_iter_max) > f2(u,x,y,n_iter_max) + beta * gamma * np.dot(gradf2(u,x,y,p,n_iter_max),u_ - u)):
                gamma = gamma * a
                u_ = proximal_g(u - gamma *gradf2(u,x,y,p,n_iter_max),gamma,p,alpha)

            u = proximal_g(u - gamma * gradf2(u,x,y,p,n_iter_max),gamma,p,alpha)
            step += 1
            delta = np.linalg.norm(gradf2(u,x,y,p,n_iter_max))
            print (delta)

        return u,step,delta

    def _grad_descent(self,x,y,w,eta,imax):
        learning_rate = self.learning_rate_
        thresh = self.tol_
        n_iter = imax
        alpha = self.alpha_
        C = self.C_
        n_iter_max = x.shape[0]
        ws = w.copy()
        k = 0
        eta0 = 1
        if(self.loss_ == 'log'):
            while(k < n_iter):
                step = 0
                loss,grad = log_loss(alpha,ws,x[step],y[step],self.weight_,C)
                grad_dir = - grad
                delta = np.inf
                gamma = 1. / (1 + alpha * step)

                while (step < n_iter_max-1):
                    #grad0 = grad
                    ws = ws + gamma*grad_dir
                    step += 1
                    gamma = 1. / (1 + alpha * step)
                    # gamma = self._line_search(ws,x[step],y[step],gamma,grad_dir)
                    loss,grad = log_loss(alpha,ws,x[step],y[step],self.weight_,C)
                    if(loss == np.inf or np.linalg.norm(grad) == np.nan):
                        step += 1
                        gamma = 1. / (1 + alpha * step)
                        # gamma = self._line_search(ws,x[step],y[step],gamma,grad_dir)
                        loss,grad = log_loss(alpha,ws,x[step],y[step],self.weight_,C)

                    #grad_dir = norm(grad)**2 / norm(grad0)**2 * grad_dir - grad
                    grad_dir = -grad
                    delta = np.linalg.norm(grad_dir)
                k += 1

        elif(self.loss_ == 'hinge'):
            while(k < n_iter):
                step = 0
                loss,grad = hinge_loss(alpha,ws,x[step],y[step],self.weight_,C)
                grad_dir = - grad
                delta = np.inf
                gamma = eta0 / (1 + alpha * eta0*step)
                while (step < n_iter_max-1):
                    #grad0 = grad
                    ws = ws + gamma*grad_dir
                    step += 1
                    gamma = eta0 / (1 + alpha*eta0*step)
                    #gamma = self._line_search(ws,x[step],y[step],gamma,grad_dir)
                    loss,grad = hinge_loss(alpha,ws,x[step],y[step],self.weight_,C)
                    if(loss == np.inf or np.linalg.norm(grad) == np.nan):
                        step += 1
                        gamma = eta0 / (1 + alpha * eta0*step)
                        #gamma = self._line_search(ws,x[step],y[step],gamma,grad_dir)
                        loss,grad = hinge_loss(alpha,ws,x[step],y[step],self.weight_,C)

                    #grad_dir = norm(grad)**2 / norm(grad0)**2 * grad_dir - grad
                    grad_dir = -grad
                    delta = np.linalg.norm(grad_dir)
                k +=1

        return ws,step,delta

    def _line_search(self,w,x,y,gamma,d):
        a = 0.5
        b = 2*gamma
        l = 0
        alpha = self.alpha_
        w = w.copy()
        x = w.copy()
        w_ = w + b*d
        while (hinge_loss(alpha,w_,x,y,self.weight_,1)[0] >hinge_loss(alpha,w,x,y,self.weight_,1)[0] + d.dot(w_-w) + 1. / (2*b*a**l) *np.linalg.norm(w-w_)**2):
            l += 1
            w_ = w + b*d

        return b*a**l

    def _evaluateEta(self,x, y, eta,w):
        step = 0
        alpha = self.alpha_
        C = self.C_
        n_iter_max = x.shape[0]
        loss,grad = hinge_loss(alpha,w,x[step],y[step],self.weight_,C)
        grad_dir = - grad
        delta = np.inf
        gamma = eta / (1 + alpha * eta*step)
        while (step < n_iter_max-1):
            #grad0 = grad
            print(loss)
            w = w + gamma*grad_dir
            step += 1
            gamma = eta / (1 + alpha*eta*step)
            #gamma = self._line_search(w,x[step],y[step],gamma,grad_dir)
            loss,grad = hinge_loss(alpha,w,x[step],y[step],self.weight_,C)
            if(loss == np.inf or np.linalg.norm(grad) == np.nan):
                step += 1
                gamma = eta / (1 + alpha * eta*step)
                #gamma = self._line_search(w,x[step],y[step],gamma,grad_dir)
                loss,grad = hinge_loss(alpha,w,x[step],y[step],self.weight_,C)

            #grad_dir = norm(grad)**2 / norm(grad0)**2 * grad_dir - grad
            grad_dir = -grad
            delta = np.linalg.norm(grad_dir)
        loss = 0
        cost = 0
        nerr = 0
        l = self.alpha_
        loss = np.dot(x,w).sum()
        loss = loss / x.shape[0]
        cost = loss + l * np.linalg.norm(w)**2
        return cost

    def _determineEta0(self, X, y,w):
      factor = 2
      loEta = 1
      print('ok')
      loCost = self._evaluateEta(X,y,loEta,w)
      hiEta = loEta * factor
      hiCost = self._evaluateEta(X ,y ,hiEta,w)
      if (loCost < hiCost):
        print('ok')
        while (loCost < hiCost):

            hiEta = loEta
            hiCost = loCost
            loEta = hiEta / factor
            loCost = self._evaluateEta(X,y, loEta,w)
            print(loCost - hiEta)

      elif (hiCost < loCost):
        while (hiCost < loCost):

            loEta = hiEta
            loCost = hiCost
            hiEta = loEta * factor
            hiCost = self._evaluateEta(X,y, hiEta,w)
            print(loCost - hiCost)

      return loEta

def log_loss(alpha,w,x,y,weight,C):
    """ evaluates log loss and its gradient at w
        rows of x are data points
    y is a vector of labels
    """
    loss = 0.0
    grad = np.zeros((w.shape[0],))
    if(y == -1):
        v = -y * w.dot(x)
        loss =  C * weight[0] * np.log(1 + np.exp(v)) + alpha * np.linalg.norm(w[:-1])**2
        w[-1] = 0
        grad = - C * weight[0] * y * x * np.exp(v) / (1 + np.exp(v)) + 2 * alpha * w
    elif(y == 1):
        v = -y * w.dot(x)
        loss = C * weight[1] * np.log(1 + np.exp(v)) + alpha * np.linalg.norm(w[:-1])**2
        w[-1] = 0
        grad = -  C * weight[1] *y * x * np.exp(v) / (1 + np.exp(v)) + 2 * alpha * w

    return loss,grad

def hinge_loss(alpha,w,x,y,weight,C):
    ws = w.copy()
    loss = 0.0
    grad = np.zeros((ws.shape[0],))
    v = y * ws.dot(x)
    ws[-1] = 0
    if(y == -1):
        loss =  0 if v > 1 else C* (1-v)
        loss += np.abs(ws).sum()
        grad = 0 if v>1 else - C* y*x
        grad += alpha * np.sign(ws)
    elif(y == 1):
        loss =  0 if v > 1 else C*( 1-v)
        loss += np.abs(ws).sum()
        grad = 0 if v>1 else - C * y*x
        grad+= alpha * np.sign(ws)

    return loss,grad

def f2(w,X,y,n):
    value = 0
    for i in range(n):
        ee = np.exp(-y[i]*(X[i,:].dot(w)))
        value = value + 1./ n * np.log(1 + ee)
    return value

def gradf2(w,X,y,p,n):
    value = np.zeros((p+1,1))

    for i in range(0,n):
        ee = np.exp(-y[i]*(X[i,:].dot(w)))
        value[:-1] =value[:-1] + 1. / n * (-y[i]) * X[i,:-1].reshape(p,1)* ee / (1 + ee)
        value[-1] = value[-1] + 1. / n * (-y[i]) * ee / (1 + ee)
    return value.ravel()

def proximal_g(w0,gamma,p,rho):
    w = np.zeros((p+1,))
    w[-1] = 0
    for i in range(0,p):
        if abs(w0[i])< gamma * rho:
            w[i] = w[-1]
        elif w0[i]>gamma * rho:
            w[i] = w0[i] - gamma *rho
        elif w0[i] < - gamma * rho:
            w[i] = w0[i] + gamma *rho
    return w
