#!/usr/bin/python2

import numpy as np
from matplotlib import pyplot as plt
from utils import proc, nvec, rmat
from multicomet import MultiCometAnimation

def project2d(vs, ns):
    # vs = vectors
    # ns = normal vectors
    Rs = [rmat(n, np.asarray([0,0,1])) for n in ns]
    return np.asarray([R.dot(v) for (R,v) in zip(Rs,vs)], dtype=np.float32)

def plot(time, pos, vel):
    # remove initial point

    # define pivot
    a0 = np.argmin(pos[:,1])
    pos -= pos[a0]
    px = np.mean(pos[:,0])
    py = np.min(pos[:,1]) + 343
    pivot = (px,py)
    #pivot = pos[a0] + [0, 343]

    x, y = pos.T
    dx = x - pivot[0]
    dy = pivot[1] - y
    angle = np.arctan2(dx, dy)
    angle = np.rad2deg(angle)
    plt.plot(time,angle)
    plt.xlabel('time (s)')
    plt.ylabel(r'$\theta (deg)$')
    plt.title('Single Pendulum With Escapement')
    plt.grid()
    plt.show()

    # show animation ...
    # data = pos[:, np.newaxis, ...]
    # fig = plt.figure()
    # ani = MultiCometAnimation(
    #         data,
    #         ['pos'],
    #         fig=fig
    #         )
    # plt.show()

    #k = np.squeeze(np.asarray(k))
    #for c in k:
    #    if np.shape(c) != (1,3):
    #        print c
    return

def process(f, opts):
    fps = 60.
    dt = 1.0 / fps
    k = f.readlines()[1:]
    k = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    print 'Length of Data : ', len(k)
    #k = k[40000:42000]

    ids = [int(k[0][0][0])]
    pos, vel = proc(k, ids, dt)

    # skip initial instability
    pos = pos[100:]
    vel = vel[100:]

    pos = np.squeeze(pos, 1) #N, 3
    vel = np.squeeze(vel, 1) #N, 3

    # flip y
    np.negative(pos[:,1], pos[:,1])
    np.negative(vel[:,1], vel[:,1])

    # project to 2d
    ref_idx = np.random.choice(len(pos), 1000, replace=False)
    ref = pos[ref_idx] # sample points to fit plane
    n = nvec(ref) # fit plane from motion, etc.
    print 'normal vector', n
    R = rmat(n, [0,0,1])
    print R.dot(n)
    pos = R.dot(pos.T).T
    vel = R.dot(vel.T).T

    pos = pos[:,:2]
    vel = vel[:,:2]
    time = np.linspace(0, len(pos)*dt, num=len(pos))
    time = time[:, np.newaxis]

    np.savetxt(
            'data.csv',
            np.concatenate((time,pos,vel), axis=-1),
            delimiter=',',
            header = 't x y vx vy'
            )

    data = np.loadtxt('data.csv', delimiter=',')

    time = data[:, 0]
    pos = data[:,1:3]
    vel = data[:,3:5]
    plot(time, pos, vel)

if __name__ == "__main__":
    #with open('build/data_single_with_escapement.txt') as f:
    #    process(f, None)
    data = np.loadtxt('data.csv', delimiter=',')
    time = data[:, 0]
    pos = data[:,1:3]
    vel = data[:,3:5]
    plot(time, pos, vel)
