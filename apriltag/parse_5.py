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

def pivot(pts):
    x, y = pts[:,:2].T
    px = np.mean(x)
    py = np.min(y) + 343
    return (px,py)

def get_angle(pos, pivot):
    x, y = pos.T
    dx = x - pivot[0]
    dy = pivot[1] - y
    angle = np.arctan2(dx, dy)
    angle = np.rad2deg(angle)
    return angle

def plot(time, data_left, data_right):
    # remove initial point

    thl, thr =  data_left[:,-1], data_right[:,-1]
    delta = (thl - thr) / 2.0
    sigma = (thl + thr) / 2.0

    fig = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')

    ax1.plot(time, delta)
    ax1.plot(time, sigma)

    ax2.plot(time, thl)
    ax2.plot(time, thr)

    #plt.legend(['delta','sigma'])
    plt.show()

    ## define pivot
    #a0 = np.argmin(pos[:,1])
    #pos -= pos[a0]
    #px = np.mean(pos[:,0])
    #py = np.min(pos[:,1]) + 343
    #pivot = (px,py)
    ##pivot = pos[a0] + [0, 343]

    #x, y = pos.T
    #dx = x - pivot[0]
    #dy = pivot[1] - y
    #angle = np.arctan2(dx, dy)
    #angle = np.rad2deg(angle)
    #plt.plot(time,angle)
    #plt.xlabel('time (s)')
    #plt.ylabel(r'$\theta (deg)$')
    #plt.title('Single Pendulum With Escapement')
    #plt.grid()
    #plt.show()

    ## show animation ...
    ## data = pos[:, np.newaxis, ...]
    ## fig = plt.figure()
    ## ani = MultiCometAnimation(
    ##         data,
    ##         ['pos'],
    ##         fig=fig
    ##         )
    ## plt.show()

    ##k = np.squeeze(np.asarray(k))
    ##for c in k:
    ##    if np.shape(c) != (1,3):
    ##        print c
    return

def process(f, opts):
    fps = 60.
    dt = 1.0 / fps
    k = f.readlines()[1:]
    k = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    print 'Length of Data : ', len(k)

    # cutoff
    #k = k[1000:2000]
    ids = np.int32([l[0] for l in k[0]]) # say 15, 25, 31

    pos, vel = proc(k, ids, dt)

    # skip initial instability
    pos = pos[100:]
    vel = vel[100:]


    #pos = np.squeeze(pos, 1) #N, 3
    #vel = np.squeeze(vel, 1) #N, 3

    # flip y
    np.negative(pos[...,1], pos[..., 1])
    np.negative(vel[...,1], vel[..., 1])

    # remove drift over time ...
    ctr = np.mean(pos[:,1:4,:], axis=1, keepdims=True) # center of three triangles
    pos -= ctr

    # calculate normal vector
    v1 = np.subtract(pos[:,2], pos[:,1])
    v2 = np.subtract(pos[:,3], pos[:,1])
    n = np.cross(v2,v1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    #print 'normal vector', n

    # align all to normal plane
    for i in range(4):
        pos[:,i] = project2d(pos[:,i], n)
        vel[:,i] = project2d(vel[:,i], n)

    # remove z component
    pos = pos[..., :2]
    vel = vel[..., :2]

    pv_l = pivot(pos[:,0])
    th_l = get_angle(pos[:,0], pv_l)

    pv_r = pivot(pos[:,4])
    th_r = get_angle(pos[:,4], pv_r)

    time = dt * np.arange(len(pos))

    nax = np.newaxis
    data = np.concatenate((
        time[...,nax],
        pos[:,0], vel[:,0], th_l[...,nax], # left mvmt
        pos[:,4], vel[:,4], th_r[...,nax] # right mvmt
        ), axis=-1)

    np.savetxt(
            'data_double.csv',
            data,
            delimiter=',',
            header='t xl yl vxl vyl thl xr yr vxr vyr thr'
            )
            
    #plt.plot(time, th_l)
    #plt.plot(time, th_r)

    #for i in range(5):
    #    plt.plot(pos[:,i,0], pos[:,i,1])
    #plt.legend(['1','2','3','4','5'])

    #bob_left = pos[:,0]
    #bob_right = pos[:,4]
    #rig_lu = pos[:,1]
    #rig_ru = pos[:,2]
    #rig_mb = pos[:,3]

    #plt.show()

    ## project to 2d
    #ref_idx = np.random.choice(len(pos), 1000, replace=False)
    #ref = pos[ref_idx] # sample points to fit plane
    #n = nvec(ref) # fit plane from motion, etc.
    #print 'normal vector', n
    #R = rmat(n, [0,0,1])
    #print R.dot(n)
    #pos = R.dot(pos.T).T
    #vel = R.dot(vel.T).T

    #time = np.linspace(0, len(pos)*dt, num=len(pos))
    #time = time[:, np.newaxis]

    #np.savetxt(
    #        'data.csv',
    #        np.concatenate((time,pos,vel), axis=-1),
    #        delimiter=',',
    #        header = 't x y vx vy'
    #        )

    #data = np.loadtxt('data.csv', delimiter=',')

    #time = data[:, 0]
    #pos = data[:,1:3]
    #vel = data[:,3:5]
    #plot(time, pos, vel)

if __name__ == "__main__":
    #with open('build/data_double_part_1.txt') as f:
    #    process(f, None)
    data = np.loadtxt('data_double.csv', delimiter=',')
    print np.shape(data)
    time = data[:, 0]
    data_left = data[:,1:6] #x,y,vx,vy,t
    data_right = data[:,6:] #x,y,vx,vy,t
    plot(time, data_left, data_right)
