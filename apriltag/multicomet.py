from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import TimedAnimation
import numpy as np
 
class MultiCometAnimation(TimedAnimation):
    def __init__(self, data, legend, fig=None, **kwargs):
        # fig setup ... 
        if not fig:
            fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
 
        # data setup ...
        self._data = data
        n, m = np.shape(data)[:2]
        self._n = n
        self._m = m
         
        xmin = np.min(data[:,:,0])
        xmax = np.max(data[:,:,0])
        xmid = (xmax+xmin)/2.
        ymin = np.min(data[:,:,1])
        ymax = np.max(data[:,:,1])
        ymid = (ymax+ymin)/2.

        #s = 2*max(np.std(data[:,:,0]), np.std(data[:,:,1]))
        #print 's', s
        s = 1.1 * max(xmax-xmin, ymax-ymin)
        # circle radius
        r = 0.01 * s
 
        # plots setup ...
        self._ls = []
        self._cs = []
        for i in range(m):
            col = np.random.uniform(size=3)
            cir = plt.Circle((0,0), r, fc=col)
            ln = Line2D([],[], color=col)
            self._cs.append(cir)
            self._ls.append(ln)
            ax.add_patch(cir)
            ax.add_line(ln)
 
        ax.set_aspect('equal')
        ax.set_xlim(xmid-s/2., xmid+s/2.)
        ax.set_ylim(ymid-s/2., ymid+s/2.)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(legend)
 
        TimedAnimation.__init__(self, fig, **kwargs)
 
    def _draw_frame(self, k):
        for i in range(self._m):
            x,y = self._data[:k, i, :2].T
            self._ls[i].set_data(x,y)
            self._cs[i].center = x[-1], y[-1]
        self._drawn_artists = self._ls + self._cs
 
    def new_frame_seq(self):
        return iter(range(1, self._n))
 
    def _init_draw(self):
        pass
