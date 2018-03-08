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

def get_pivot(pts):
    n,m = np.shape(pts)
    if n == 2:
        pts = np.transpose(pts)
        n,m = np.shape(pts)

    x, y = np.transpose(pts)
    sx, sy = np.sum(pts, axis=0)
    sxx, syy = np.sum(np.square(pts), axis=0)

    d11 = n * np.sum(x*y) - sx*sy
    d20 = n * sxx - sx*sx
    d02 = n * syy - sy*sy
    d30 = n * np.sum(x*x*x) - sxx*sx
    d03 = n * np.sum(y*y*y) - syy*sy
    d21 = n * np.sum(x*x*y) - sxx*sy
    d12 = n * np.sum(y*y*x) - syy*sx

    x = ((d30 + d12) * d02 - (d03 + d21) * d11) / (2 * (d20 * d02 - d11 * d11))
    y = ((d03 + d21) * d20 - (d30 + d12) * d11) / (2 * (d20 * d02 - d11 * d11))

    c = (sxx+syy - 2*x*sx - 2*y*sy) / n
    r = np.sqrt(c + x*x+y*y)

    return (x,y), r

def get_pivot_2(pts, r):
    # r = 343
    pts = np.transpose(pts)
    n = len(pts)
    res = []
    for i in range(n):
        for j in range(i+1,n):
            p1, p2 = pts[i], pts[j]
            dp = np.subtract(p2,p1)
            d = np.linalg.norm(dp)
            a = d/2.
            h = np.sqrt(np.abs(r**2-a**2))
            if dp[0] >= 0:
                x3 = p2[0] - h * dp[1] / d
                y3 = p2[1] + h * dp[0] / d
            else:
                x3 = p2[0] + h * dp[1] / d
                y3 = p2[1] - h * dp[0] / d
            res.append((x3,y3))
    return np.mean(res, axis=0)


def circle(c, r):
    th = np.linspace(-np.pi, np.pi)
    x = c[0] + r*np.cos(th)
    y = c[1] + r*np.sin(th)
    plt.plot(x,y)

def main(f, opts):
    k = f.readlines()[1:]
    k = [np.asarray([r.split(',')[1:] for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    k = np.squeeze(np.asarray(k))

    # project to 2d
    ref = k[-300:] # ~ 5 sec
    nv = nvec(ref)
    R = rmat(nv, [0,0,1])
    v = np.asarray([R.dot(v) for v in k])
    v -= v[0] # normalize ...
    v[:,1] = -v[:,1]# flip y

    x, y = v[300:,:2].T

    mxx = np.max(x)
    mnx = np.min(x)
    mdx = (mxx+mnx)/2.
    mxy = np.max(y)
    mny = np.min(y)
    mdy = (mxy+mny)/2.
    scale = max(mxx-mnx, mxy-mny)
    xlim = [mdx-scale/2, mdx+scale/2]
    ylim = [mdy-scale/2, mdy+scale/2]

    pivot = [0, y[0] + 343.] # 34.3 cm = 343mm
    pivot2, radius = get_pivot([x[500:-500],y[500:-500]])
    pivot3 =  get_pivot_2([x[-1000:-500], y[-1000:-500]], 343.)
    print 'ps', pivot, pivot2, pivot3

    pivot = pivot3

    dx = x - pivot[0]
    dy = pivot[1] - y
    th = np.arctan2(dx, dy)
    print len(th)

    fig = plt.figure()
    ax = fig.add_subplot(2,2,3)
    ax2 = fig.add_subplot(2,2,4)
    ax3 = fig.add_subplot(2,1,1)

    ax.plot(x,y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_title('x-y equal scale')

    ax2.plot(x,y)
    ax2.set_xlabel('x(mm)')
    ax2.set_ylabel('y(mm)')
    ax2.set_title('x-y fit')

    ax3.plot(np.arange(len(th))/60., np.rad2deg(th)) # assume 60 fps
    ax3.set_xlabel('t(s)')
    ax3.set_ylabel(r'$\theta (deg)$')
    ax3.set_title('Single Pendulum Behavior, With Escapement')
    #circle(pivot, 343.)
    #circle(pivot2, radius)
    #plt.legend(['data','pg','pc'])
    #plt.plot(np.arange(len(th))/60., th)
    plt.show()

    # calculate normal vector based on answer [here](https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points)

if __name__ == "__main__":
    with open('build/data_single_escapement.txt') as f:
        main(f, None)
