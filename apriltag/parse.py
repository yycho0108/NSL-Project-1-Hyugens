import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, TimedAnimation, writers


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

def v2vx(v):
    v1,v2,v3 = v
    return np.reshape([0, -v3, v2, v3, 0, -v1, -v2, v1, 0], (3,3)).astype(np.float32)

def rmat(v0, v1):
    u = v0
    v = v1

    #print np.linalg.norm(u)
    #print np.linalg.norm(v)
    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    ax,ay,az = axis
    th = np.arccos(np.dot(u, v))
    c = np.cos(th)
    _c = 1.0 - c
    s = np.sin(th)

    M = (1.0 - c) * np.outer(axis, axis)
    M += np.reshape([c, -s*az, s*ay, s*az, c, -s*ax, -s*ay, s*ax, c],(3,3))
    #print u, np.dot(M, u)

    #M = [
    #        [ax*ax*_c+c, ax*ay*_c-s*az, ax*az*_c+s*ay],
    #        [ax*ay*_c+s*az, ay*ay*_c+c, ay*az*_c-s*ax],
    #        [ax*az*_c-s*ay, ay*az*_c+s*ax, az*az*_c+c]
    #        ]
    #print M - M0

    #print u, np.asarray(M).dot(u[:,np.newaxis])
    # M * v0 = v1
    return np.asarray(M, dtype=np.float32)


def project2d(vs, ns):
    # vs = vectors
    # ns = normal vectors
    Rs = [rmat(n, np.asarray([0,0,1])) for n in ns]
    return np.asarray([R.dot(v) for (R,v) in zip(Rs,vs)], dtype=np.float32)


class SubplotAnimation(TimedAnimation):
    def __init__(self, k2, t):
        fig = plt.figure()
        fig.set_dpi(100)
        fig.set_size_inches(12, 6)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)

        self.k2 = k2
        self.t = t

        x1, y1 = k2[:,0,:2].T
        x2, y2 = k2[:,1,:2].T
        x3, y3 = k2[:,2,:2].T
        x4, y4 = k2[:,3,:2].T

        self.x1, self.y1 = x1,y1
        self.x2, self.y2 = x2,y2

        self.th1 = np.arctan2(x1-x3,(y3-25.4)-y1)
        self.th2 = np.arctan2(x2-x4,(y4-25.4)-y2)

        # ax1 
        ax1.set_xlabel('x(mm)')
        ax1.set_ylabel('y(mm)')
        ax1.set_aspect('equal', 'datalim')
        self.ps1 = [Line2D([],[]) for _ in range(6)]

        self.pc1 = plt.Circle((0,0), 10, fc='r')
        self.pc2 = plt.Circle((0,0), 10, fc='b')

        [ax1.add_line(l) for l in self.ps1]
        ax1.add_patch(self.pc1)
        ax1.add_patch(self.pc2)

        ax1.set_xlim(np.min(k2[:,:,0]), np.max(k2[:,:,0]))
        ax1.set_ylim(np.min(k2[:,:,1]), np.max(k2[:,:,1]))
        ax1.set_aspect('equal','datalim')

        # ax2
        #ax2.set_xlabel('t(sec)')
        ax2.set_ylabel(r'$\theta_l$(rad)')
        self.p2 = Line2D([],[])
        ax2.add_line(self.p2)
        ax2.set_xlim(np.min(t), np.max(t))
        ax2.set_ylim(np.min(self.th1), np.max(self.th1))
        plt.setp(ax2.get_xticklabels(), visible=False)
        self.ax2 = ax2

        # ax3
        ax3.set_xlabel('t(sec)')
        ax3.set_ylabel(r'$\theta_r$(rad)')
        self.p3 = Line2D([],[])
        ax3.add_line(self.p3)
        ax3.set_xlim(np.min(t), np.max(t))
        ax3.set_ylim(np.min(self.th2), np.max(self.th2))
        self.ax3 = ax3

        TimedAnimation.__init__(self, fig, interval=1000/60., blit=True, repeat=False)

    def _draw_frame(self, n):
        for i in range(6):
            (x,y) = self.k2[:n,i,:2].T
            self.ps1[i].set_data(x,y)

        self.pc1.center = self.x1[n], self.y1[n]
        self.pc2.center = self.x2[n], self.y2[n]

        i0 = max(0, int(n - 0.5*60))
        i1 = min(len(self.t), int(n + 0.5*60))

        self.p2.set_data(self.t[i0:i1], self.th1[i0:i1])
        self.ax2.set_xlim(min(self.t[i0:i1]), max(self.t[i0:i1]))
        self.ax2.set_ylim(min(self.th1[i0:i1]), max(self.th1[i0:i1]))

        self.p3.set_data(self.t[i0:i1], self.th2[i0:i1])
        self.ax3.set_xlim(min(self.t[i0:i1]), max(self.t[i0:i1]))
        self.ax3.set_ylim(min(self.th2[i0:i1]), max(self.th2[i0:i1]))
        
        self._drawn_artists = self.ps1 + [self.p2, self.p3] + [self.pc1, self.pc2]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = self.ps1 + [self.p2, self.p3]
        for l in lines:
            l.set_data([], [])

with open('data_3.txt', 'r') as f:

    # flags
    dup_id = False
    anim = False
    save = True

    fps = 60

    k = f.readlines()[1:]

    k = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    #k = [id,x,y,z,id,x,y,z,id,x,y,z]

    # process & match ids, etc...
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
    np.negative(k2[:,:,1], k2[:,:,1]) # flip y

    t = (1./fps) * np.arange(len(k2)) # 60fps


    # 3-4-5 : defines a plane
    x3, y3, z3 = k2[:,2].T
    x4, y4, z4 = k2[:,3].T
    x5, y5, z5 = k2[:,4].T

    v1 = np.subtract(k2[:,3], k2[:,2])
    v2 = np.subtract(k2[:,3], k2[:,4])
    n = np.cross(v1,v2)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # align to normal plane
    for i in range(6):
        k2[:,i] = project2d(k2[:,i], n)
    k2 -= k2[:,np.newaxis, 5] # subtract track reference, for camera motion

    # align to horizontal
    v1 = np.subtract(k2[:,3], k2[:,2])
    v1[:,2] = 0 # set z=0 to force rotation in z axis
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
    Rs = [rmat(_n, np.asarray([1,0,0])) for _n in v1]
    for i in range(6):
        k2[:,i] = np.asarray([R.dot(v) for (R,v) in zip(Rs,k2[:,i])], dtype=np.float32)
    k2 -= k2[:,np.newaxis, 5] # subtract track reference, for camera motion

    # convert to 2d coords for each
    x1, y1 = k2[:,0,:2].T
    x2, y2 = k2[:,1,:2].T

    x3, y3 = k2[:,2,:2].T
    x4, y4 = k2[:,3,:2].T
    x5, y5 = k2[:,4,:2].T

    x6, y6 = k2[:,5,:2].T

    th1 = np.arctan2(x1-x3,(y3-25.4)-y1)
    th2 = np.arctan2(x2-x4,(y4-25.4)-y2)
    #plt.plot(t,th1)
    #plt.show()
    #sys.exit(0)

    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if anim:
        dx2, dy2 = x2-x1, y2-y1
        dx3, dy3 = x3-x1, y3-y1

        #plt.plot(t, -(y2-y1))
        fig = plt.figure()
        fig.set_dpi(100)
        fig.set_size_inches(7, 6.5)
        ax = fig.gca()

        p1, = ax.plot([], [])
        p2, = ax.plot([], [])

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

        Writer = writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
        
        ani = FuncAnimation(fig, update, frames=t,
                        init_func=init, blit=True, interval=1000./60, repeat=False)
        #ani.save('meh.gif', writer='imagemagick')
        
            
        #plt.plot(t,x2-x1)
        #plt.plot(t,x3-x1)

        #plt.plot(x3,y3) # tag 1 , x
    else:

        #fig = plt.figure()
        #fig.set_dpi(100)
        #fig.set_size_inches(6,8)

        #ax1 = fig.gca()
        ##ax1 = fig.add_subplot(1,2,1)
        ##ax2 = fig.add_subplot(1,2,2)

        #p1, = ax1.plot(x1,y1)
        #p2, = ax1.plot(x2,y2)
        #p3, = ax1.plot(x3,y3)
        #p4, = ax1.plot(x4,y4)
        #p5, = ax1.plot(x5,y5)
        #p6, = ax1.plot(x6,y6)

        #pc1 = plt.Circle((0, 0), 10, fc='r')
        #pc2 = plt.Circle((0, 0), 10, fc='b')
        #pc3 = plt.Circle((0, 0), 5, fc='g')

        #plt.axes().set_aspect('equal', 'datalim')
        #plt.axis('equal')
        #plt.xlabel('x(mm)')
        #plt.ylabel('y(mm)')
        #plt.title('System data plot')
        #plt.legend(['1','2','3','4','5','6'])

        ###plt.plot(t, -(y2-y1))


        #def init():
        #    #plt.ylim([-70, 0])
        #    #plt.xlim([-200,200])
        #    ax1.add_patch(pc1)
        #    ax1.add_patch(pc2)
        #    ax1.add_patch(pc3)
        #    return [p1,p2,p3,p4,p5,p6,pc1,pc2,pc3]

        #def update(n):
        #    p1.set_data(x1[:n], y1[:n])
        #    p2.set_data(x2[:n], y2[:n])
        #    p3.set_data(x3[:n], y3[:n])
        #    p4.set_data(x4[:n], y4[:n])
        #    p5.set_data(x5[:n], y5[:n])
        #    p6.set_data(x6[:n], y6[:n])

        #    pc1.center = x1[n], y1[n]
        #    pc2.center = x2[n], y2[n]
        #    pc3.center = x5[n], y5[n]
        #    #pc3.center = np.mean([
        #    #            [x3[n], y3[n]],
        #    #            [x4[n], y4[n]],
        #    #            [x5[n], y5[n]],
        #    #            ], axis=0)

        #    
        #    return [p1,p2,p3,p4,p5,p6,pc1,pc2,pc3]
        #ani = FuncAnimation(fig, update, frames=range(len(t)),
        #                init_func=init, blit=True, interval=1000./fps, repeat=False)

        ani = SubplotAnimation(k2,t)
        if save:
            Writer = writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
            ani.save('meh3.mp4', writer=writer)
    plt.show()

