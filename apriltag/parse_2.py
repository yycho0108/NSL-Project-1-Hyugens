import numpy as np
from matplotlib import pyplot as plt

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


def project2d(vs, ns):
    # vs = vectors
    # ns = normal vectors
    Rs = [rmat(n, np.asarray([0,0,1])) for n in ns]
    return np.asarray([R.dot(v) for (R,v) in zip(Rs,vs)], dtype=np.float32)

def nvec(pts):
    shape = np.shape(pts)
    if not shape[0] == 3:
        pts = pts.T
    # now of shape (3,N)
    c = np.mean(pts, axis=1, keepdims=True)
    pts = np.subtract(pts, c)
    u,s,v = np.linalg.svd(pts, compute_uv=True)
    return u[:,-1] / np.linalg.norm(u[:,-1])

def main(f, opts):
    k = f.readlines()[1:]
    k = [np.asarray([r.split(',')[1:] for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    k = np.squeeze(np.asarray(k))

    # project to 2d
    ref = k[-300:] # ~ 5 sec
    nv = nvec(ref)
    R = rmat(nv, [0,0,1])
    v = np.asarray([R.dot(v) for v in k])
    v -= v[0]
    v_2d = v[:,:2]

    x, y = v_2d.T
    y = -y

    pivot = y+343. # 34.3 cm = 343mm

    dx = x
    dy = pivot - y
    th = np.arctan2(dx, dy)
    print len(th)

    #plt.plot(x,y)
    plt.plot(th)
    plt.show()

    # calculate normal vector based on answer [here](https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points)

if __name__ == "__main__":
    with open('build/data_single_escapement.txt') as f:
        main(f, None)
