import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSS
import sys

def hx(x):
    #x,y,vx,vy
    return x[:3]

def fx(x, dt):
    x,y,z,vx,vy,vz = x
    x=x+vx*dt
    y=y+vy*dt
    z+y+vz*dt
    return (x,y,z,vx,vy,vz)

def reorder(e1, id0):
    # doesn't handle duplicate ids anymore.

    id1 = np.int32([k[0] for k in e1]) # say 31, _, 25, need to be organized as [?, 0, 2]

    midx = []
    idx = []
    res = []
    for i0, _id0 in enumerate(id0):
        for i1, _id1 in enumerate(id1):
            if _id0==_id1:
                res.append(e1[i1][1:]) # skip id
                break
        else:
            # id not found
            res.append(None)
            midx.append(i0)
    return np.int32(midx), res

def proc(raw_data, ids, dt):
    # Process & match ids, etc...

    n = len(ids)

    fpt = MSS(6, 1e-3, 2, 0.0)
    ukfs = [UKF(6,3,dt,hx=hx,fx=fx,points=fpt) for _ in range(n)]
    for i, u in enumerate(ukfs):
        u.x = np.zeros(6, dtype=np.float32)
        u.x[:3] = raw_data[0][i][1:]
        u.P = np.square(np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])) # initial cov
        u.R = 0.005 * np.eye(3) #(5.0)**2 * np.eye(3) # measurement cov, ~ +-5mm std.
        u.Q = (0.3)**2 * np.eye(6) # process cov ~ +-3mm std.

    l = len(raw_data) - 1 # skip the first one, so...
    pos = []
    vel = []
    for c, e in enumerate(raw_data[1:]):
        if c % 100 == 0:
            # log progress
            log_string = '{}/{}\r'.format(c, l)
            sys.stdout.write(log_string)
            sys.stdout.flush()
        [u.predict(dt) for u in ukfs]
        midx, obs = reorder(e, ids)
        if midx:
            # only update applicable ones
            for i in range(n):
                if i in midx:
                    continue
                ukfs[i].update(obs[i])
        else:
            # correct based on estimates
            [u.update(e) for (u,e) in zip(ukfs, obs)]

        entry = []
        ventry = []
        for u in ukfs:      
            entry.append(u.x[:3])
            ventry.append(u.x[3:])
        pos.append(np.float32(entry))
        vel.append(np.float32(ventry))

    vel = np.float32(vel)
    pos = np.float32(pos)
    return pos, vel

def nvec(pts):
    shape = np.shape(pts)
    if not shape[0] == 3:
        pts = pts.T
    # now of shape (3,N)
    c = np.mean(pts, axis=1, keepdims=True)
    pts = np.subtract(pts, c)
    u,s,v = np.linalg.svd(pts, compute_uv=True)
    return u[:,-1] / np.linalg.norm(u[:,-1])

def rmat(v0, v1):
    # returns 3d rotation Matrix M, such that
    # M * uvec(v0) = uvec(v1)

    u = v0
    v = v1

    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    ax,ay,az = axis
    th = np.arccos(np.dot(u, v))
    c = np.cos(th)
    _c = 1.0 - c
    s = np.sin(th)

    M = (1.0 - c) * np.outer(axis, axis)
    M += np.reshape([c, -s*az, s*ay, s*az, c, -s*ax, -s*ay, s*ax, c],(3,3))
    return np.asarray(M, dtype=np.float32)
