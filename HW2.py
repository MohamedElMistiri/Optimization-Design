import numpy as np
import matplotlib.pyplot as plt

def f(x):
    obj = (-2*x[0]-3*x[1]+2)**2 + x[0]**2 + (x[1]-1)**2
    return obj

def gradient(x):
    g = np.array([10*x[0]+12*x[1]-8,
                12*x[0]+20*x[1]-14]).reshape(2,1)
    return g

def gradient_decent(x0,H,tol=10**-3,exact_line=True,t=0.5):
    x = x0
    g = gradient(x)
    counter = 0
    sol = []
    while abs(g[0]) > tol and abs(g[1]) > tol and counter < 100:
        alpha = 1
        counter = counter + 1
        if exact_line:
            alpha = np.matmul(g.T,g)/np.matmul(np.matmul(g.T,H),g)
        else:
            counter1 = 1
            phi = f(x) - t*np.matmul(g.T,g)*alpha
            while f(x-alpha*g) > phi and counter1 < 20:
                alpha = t*alpha
                counter1 = counter1  + 1
                phi = f(x) - t*np.matmul(g.T,g)*alpha
                # print(alpha)
                # print(f(x-alpha*g)> phi)
        x = x - alpha*g
        g = gradient(x)
        sol.append(f(x))
        print("Iteration {}:\nat x_2 = {} and x_3 = {}\ngradient values: {}\nobjective value = {}\n".format(counter,x[0],x[1],g.T,f(x)))
    plt.plot((np.abs(sol-sol[-1])))
    plt.yscale('log')
    plt.ylabel('log|f(k) - f*|')
    plt.xlabel("iteration (k)")
    plt.show()
    return x, sol

def newton_method(x0,H,tol=10**-3):
    x = x0
    g = gradient(x)
    counter = 0
    sol = []
    while abs(g[0]) > tol and abs(g[1]) > tol:
        counter = counter + 1
        x = x - np.linalg.inv(H)@g
        g = gradient(x)
        sol.append(f(x))
        print("Iteration {}:\nat x_2 = {} and x_3 = {}\ngradient values: {}\nobjective value = {}\n".format(counter,x[0],x[1],g.T,f(x)))
    plt.plot((np.abs(sol-sol[-1])))
    plt.yscale('log')
    plt.ylabel('|f(k) - f*|')
    plt.xlabel("iteration (k)")
    plt.show()
    return x, sol

H = np.array([[10,12],[12,20]])

x0 = np.array([-5, 50]).reshape(2,1)

x, sol1 = gradient_decent(x0,H,exact_line=False,t=0.5)
x,sol2 = newton_method(x0,H)



# print(g_x2(x))
# print(g_x3(x))
