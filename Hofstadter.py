import numpy as np
import numpy.linalg as linalg
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import fractions              # Available in Py2.6 and Py3.0
from scipy.linalg import eigvals_banded

def approx2(c, maxd):
    'Fast way using continued fractions'
    return fractions.Fraction.from_float(c).limit_denominator(maxd)


V1=1.0
V2=1.0
a = 1.0
b = 1.0
k = [0.0, 0.0]
steps = 1000.0
maxd = 1000

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 1, 1)

for alpha in np.arange(1.0/steps, 1.0, 1.0/steps):
    alpha_rational = approx2(alpha, maxd)
    p = np.float(alpha_rational.numerator)
    q = np.float(alpha_rational.denominator)
    print str(alpha) + ', ' + str(p) + '/' + str(q) 
    if p > 1:
#        A = np.zeros([np.int(p),np.int(p)], dtype=np.complex128)
#        for n in range(np.int(p)):
#            A[n,n] = 2.0*V2*np.cos(q*b*k[1]/p + 2*np.pi*np.float(n)*q/p)
#            if n < (p-1):
#                A[n,n+1] = V1*np.exp(np.complex(0.0,q*a*k[0]/p))
#            if n > 0:
#                A[n,n-1] = V1*np.exp(np.complex(0.0,-q*a*k[0]/p))
#        [d,v] = linalg.eig(A)

        A = np.zeros([2,p], dtype=np.complex128)
        for n in range(np.int(p)):
            A[1,n] = 2.0*V2*np.cos(q*b*k[1]/p + 2*np.pi*np.float(n)*q/p)
            if n < (p-1):
                A[0,n+1] = V1*np.exp(np.complex(0.0,q*a*k[0]/p))
            #if n > 0:
            #    A[2,n-1] = V1*np.exp(np.complex(0.0,-q*a*k[0]/p))
        #d = eigvals_banded(A) 
        d = eigvals_banded(A) 

        ds = np.sort(d)
        ax.plot(alpha*np.ones(p), ds, '.b',linewidth=0.5,markersize=1.0)
    
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$E$')

fig.show()


