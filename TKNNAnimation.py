import matplotlib.pyplot as plt
import numpy as np
import time
import pylab
from scipy.linalg import eig_banded
import os

# Set up system parameters
p = 5.0
q = 1.0
a = 1.0
b = 1.0
V1 = 0.001
V2 = 1.0
k1 = 0.0  # expect to be the same for all k1

# Set up program specifics
frames = 500 
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
k2start = 0.2
levels = np.int(p)
k2max = 100.0 * np.pi
k2min = -100.0 * np.pi
Display = True 
MakeMovie = False 
Prefix = 'TKNN_p_' + str(np.int(p)) + '_q_' + str(np.int(q)) 
files = list()
speed = 0.3

# Set up some arrays 
E = np.zeros(levels)
k2points = np.zeros(levels)
positions = np.zeros(levels)
ns = np.zeros(levels)
prev_ns = np.zeros(levels)
ss = np.zeros(levels)
k2values = np.arange(k2min, k2max, 0.01)
cosk2 = np.cos(k2values)


# Create a new figure 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

frame = 0
while frame < frames:
    k2 = k2start + speed*frame 

    # Create effective Hamiltonian and solve
    A = np.zeros([2,levels], dtype=np.complex128)
    for n in range(levels):
        A[1,n] = V2*np.cos(q*b*k2/p + 2*np.pi*np.float(n)*q/p)
        if n < (levels-1):
            A[0,n+1] = V1*np.exp(np.complex(0.0,q*a*k1/p))
    [d, v] = eig_banded(A) 

    for i in range(levels):
        E[i] = d[i]
        State = v[:,i]
        StateAbs = abs(State)
        ns[i] = StateAbs.argmax()
        if frame == 0:
            prev_ns[i] = ns[i]
            positions[i] = ns[i]
        if frame > 0:
            ss[i] = ns[i] - prev_ns[i]
            if ss[i] > p/2.0:
                ss[i] = -(p - ss[i])
            elif ss[i] < -p/2.0:
                ss[i] = p + ss[i]
                          
        positions[i] += ss[i] 
        k2points[i] = (q*b*k2/p + 2.0*np.pi*positions[i]*q/p) 
        prev_ns[i] = ns[i]


    if (k2points.max() > k2max) or (k2points.min() < k2min):
        break;

    ax.clear()
    ax.plot(k2values, cosk2, '-b')

    indices = E.argsort()

    for n in range(np.int(p)):
        ax.plot(k2points[indices[n]], E[indices[n]], colours[n] + 'o', markersize=10)
    
    ax.set_ylim((-1.2, 1.2))
    ax.set_xlim((k2min, k2max))
    ax.set_xlabel('x')
    ax.set_ylabel('E')
    ax.set_title('P = ' + str(np.int(p)) + ', Q = ' + str(np.int(q))) 
    if Display:
        fig.show()
        pylab.pause(0.01)

    if MakeMovie:
        fig.savefig(Prefix + '_frame_' + '{0:04d}'.format(frame) + '.png') 
        files.append(Prefix + '_frame_' + '{0:04d}'.format(frame) + '.png')

    frame += 1

if MakeMovie:
    print 'Making movie animation.mpg - this make take a while'
    os.system("mencoder 'mf://" + Prefix + "_frame_*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o " + Prefix + ".mpg")
    os.system("rm " + Prefix + "_frame_*.png")

