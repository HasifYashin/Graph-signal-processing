import numpy as np
import matplotlib.pylab as plt

def init_graph(f, W):
    summ = lambda x,y: x+y
    D = [reduce(summ, neigh) for neigh in W]
    D = np.diag(D)
    L = np.subtract(D, W)
    eigv, eigvecs= np.linalg.eig(L)
    eigvecs = np.transpose(eigvecs)
    s = np.argsort(eigv)
    eigv = eigv[s]
    eigvecs = eigvecs[s]
    f_c = [np.inner(f, eigvec) for eigvec in eigvecs]
    print "Start......."
    # print f_c ,"\n", "\n", "\n\n"
    print eigv, f_c
    
    plt.plot(eigv, f_c)
    plt.show()


f = [7, 5, 2, -1, 2, 5, -5]
W1 = [[0, 1, 0, 0, 0, 1, 0],
      [1, 0, 1, 0, 0, 0, 0],
      [0, 1, 0, 1, 0, 0, 0],
      [0, 0, 1, 0, 1, 0, 1],
      [0, 0, 0, 1, 0, 1, 0],
      [1, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0, 0, 0]]

W2 = [[0, 0, 0, 1, 0, 0, 1],
      [0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [1, 1, 0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0]]

print "Graph 1........."
init_graph(f, W1)

print "Graph 2........."
init_graph(f, W2)
