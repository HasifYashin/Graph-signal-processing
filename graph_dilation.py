import numpy as np
import matplotlib.pylab as plt
from pygsp import graphs, filters, plotting

def get_eig_val_vec_fc(f, W):
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
    return eigv, eigvecs, f_c

    
def f_after_t(f, t, eigvs, eigvecs, f_c):
    e = [np.exp(-t * lambd) for lambd in eigvs]
    fun= lambda i: reduce(lambda x,y:x+y,  [ f_c[l] * e[l] * eigvecs[l][i] for l in range(len(eigvs))])
    f_next = [fun(i) for i in range(len(f))]
    return f_next
# f = [1, 0, 0, 0]
# W1 = [[0, 2, 1, 0],
#       [2, 0, 0, 0],
#       [1, 0, 0, 1],
#       [0, 0, 1, 0]]

W2 =  np.array([[0,1,0,0,0,1,1,0,0,0,0,0,0],
      [1,0,1,0,0,0,0,1,0,0,0,0,0],
      [0,1,0,1,0,0,0,0,1,0,0,0,0],
      [0,0,1,0,1,0,0,0,0,1,0,0,0],
      [0,0,0,1,0,1,0,0,0,0,1,0,0],
      [1,0,0,0,1,0,0,0,0,0,0,1,0],
      [1,0,0,0,0,0,0,1,0,0,0,1,0],
      [0,1,0,0,0,0,1,0,1,0,0,0,0],
      [0,0,1,0,0,0,0,1,0,1,0,0,0],
      [0,0,0,1,0,0,0,0,1,0,1,0,0],
      [0,0,0,0,1,0,0,0,0,1,0,1,0],
      [0,0,0,0,0,1,1,0,0,0,1,0,0],
      [1,1,1,1,1,1,0,0,0,0,0,0,0]])


f = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
t = 2
eigvs, eigvecs, f_c = get_eig_val_vec_fc(f, W2)
f_new = f_after_t(f, t, eigvs, eigvecs, f_c)
print f, "\n", f_new


W2 = np.array(W2)
G2 = graphs.Graph(W2)
co = [[3, 7], [4,5], [4,4], [3,2], [2,4], [2,5],
      [3,8], [5,6], [5,3], [3,1], [1,3], [1,6], [3,4.5]] 
G2.set_coordinates(co)

s = np.array(f)
s1 = np.array(f_new)

plotting.plot_signal(G2, s, limits = [-1, 1])
plotting.plot_signal(G2, s1,limits = [-1, 1])







# W1 = np.array(W1)
# G1 = graphs.Graph(W1)
# G1.set_coordinates('ring2D')

# s = np.array(f)
# s1 = np.array(f_new)
# print s, s1
# o=np.zeros((N));
# s += rs.uniform(-0.05, 0.05, size=N)
# s2 = g.filter(s)

# G.set_coordinates('ring2D')
# print(s)
# print(s2)


# plotting.plot_signal(G1, s)
# plotting.plot_signal(G1, s1)


# print('{} nodes, {} edges'.format(G1.N, G1.Ne))
# G1.plot()


# tau = 1
# def g(x):
#      return 1. / (1. + tau * x)
# g = filters.Filter(G, g)

# rs = np.random.RandomState(42)
# s=np.zeros((N))
# s = np.array(f)
# s1 = np.array(f_new)
# print s, s1
# # o=np.zeros((N));
# # s += rs.uniform(-0.05, 0.05, size=N)
# # s2 = g.filter(s)

# # G.set_coordinates('ring2D')
# # print(s)
# # print(s2)


# plotting.plot_signal(G1, s)
# plotting.plot_signal(G1, s1)

# plt.plot(eigv, f_c)
# plt.show()


