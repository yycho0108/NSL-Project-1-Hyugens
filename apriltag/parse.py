import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers


class Tag(object):
    def __init__(self, data):
        self._id = data[0]
        self._x = data[1:]
    def match(self, o):
        if self._id != o._id:
            return np.inf
        else:
            return np.linalg.norm(self._x - o._x)

def reorder(e0, e1):
    t0 = [Tag(i) for i in e0]
    t1 = [Tag(i) for i in e1]

    n0 = len(t0)
    n1 = len(t1)

    cost = np.zeros((n0,n1), dtype=np.float32)
    for i in range(n0):
        for j in range(n1):
            cost[i,j] = t0[i].match(t1[j])

    #cost = np.square(np.subtract.outer(e0, e1))
    #print cost.shape
    i,j = linear_sum_assignment(cost)

    #if (n0 != 3 or n1 != 3):
    #    print cost
    #    print n0, n1
    #    print i,j

    # t0[i] => t1[j]
    return e1[j]

with open('data.txt', 'r') as f:
    k = f.readlines()[1:]
    k = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    #print reorder(k[0], k[1])
    #k = [id,x,y,z,id,x,y,z,id,x,y,z]
    e0 = k[0]
    k2 = []
    for e in k:
        e0n = reorder(e0, e)
        if len(e0n) < 3:
            # TODO: fill - with - anticipated
            continue
        e0 = e0n
        k2.append(np.copy(e0[:,1:]))
    #print len(k)
    k2 = np.asarray(k2)
    print k2.shape


    t = range(len(k2))
    x1, y1, z1 = k2[:,0].T
    x2, y2, z2 = k2[:,1].T
    x3, y3, z3 = k2[:,2].T

    dx2, dy2 = x2-x1, -(y2-y1)
    dx3, dy3 = x3-x1, -(y3-y1)


    #plt.plot(t, -(y2-y1))
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)
    ax = fig.gca()

    p1, = plt.plot([], [])
    p2, = plt.plot([], [])

    p3 = plt.Circle((0, 0), 0.75, fc='r')
    p4 = plt.Circle((0, 0), 0.75, fc='b')

    def init():
        plt.ylim([-70, 0])
        plt.xlim([-200,200])
        ax.add_patch(p3)
        ax.add_patch(p4)
        return p1, p2, p3, p4

    def update(n):
        p1.set_data(dx2[:n], dy2[:n])
        p2.set_data(dx3[:n], dy3[:n])
        p3.center = dx2[n], dy2[n]
        p4.center = dx3[n], dy3[n]
        return p1, p2, p3, p4

    #Writer = writers['ffmpeg']
    #writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    
    ani = FuncAnimation(fig, update, frames=t,
                    init_func=init, blit=True, interval=20, repeat=False)
    #ani.save('meh.gif', writer='imagemagick')
    
        
    #plt.plot(t,x2-x1)
    #plt.plot(t,x3-x1)

    #plt.plot(x3,y3) # tag 1 , x
    plt.show()
