import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]
    return im2
def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2


def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
           n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
           n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
           n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
           n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
           n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
           n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
           n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
           n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
           n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        if (i==0 and j==0):
            n=[(0,1), (1,0), (1,1)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1), (1,N-2)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0), (M-2, 1)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1), (M-2,N-2)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j), (1,j-1), (1,j+1)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j), (M-2,j-1), (M-2,j+1)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1), (i-1,1), (i+1,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2), (i-1,N-2), (i+1,N-2)]
        else:
            n = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),  (i + 1, j - 1),  (i + 1, j),  (i + 1, j + 1), (i, j - 1), (i, j + 1)]
        return n
    return -1

def energy(target, neighbour_values, observed_value):
    bias = 0
    index_nei = 1
    index_xy = 2
    #(0,1,2) for tmac2, loop5 (0.7,1)
    #(0,1,2) for pug, loop6 (0.7,1)
    energy = bias * target - index_nei * target * sum(neighbour_values) - index_xy * target * (observed_value - 0.5 )
    return energy

def prob(target, neighbour_values, observed_value):
    return -energy(target, neighbour_values, observed_value)#simplify the prob for convinience

def icm(y):
    x = init(y)
    rows = len(y)
    cols = len(y[0]) #assume not empty
    PASS = 100
    for i in range(PASS):
        flag = True
        for m in range(rows):
            for n in range(cols):
                ori = x[m][n]
                nei = [x[i] for i in neighbours(m,n,rows,cols)]
                if prob(1,nei,y[m][n]) > prob(-1,nei,y[m][n]):
                    x[m][n] = 1
                else:
                    x[m][n] = -1
                if flag and (x[m][n] != ori):
                    flag = False
        if flag:
            print('stop in loop ', i)
            return x
    return x

def init(im_noise):
    thresh = 0.5
    x_im = im_noise
    x_im = np.copy(im_noise)
    for i in range(len(im_noise)):
        for j in range(len(im_noise[0])):
            if im_noise[(i,j)] > thresh:
                x_im[(i,j)] = 1
            else:
                x_im[(i,j)] = -1
    return x_im

# proportion of pixels to alter
prop = 0.7
varSigma = 1
im = imread('../pic/pug_grey.jpg')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im,cmap='gray')
im_noise = add_gaussian_noise(im,prop,varSigma)
#im_noise = add_saltnpeppar_noise(im,prop)
ax2 = fig.add_subplot(132)
ax2.imshow(im_noise,cmap='gray')
x_im = icm(im_noise)
ax3 = fig.add_subplot(133)
ax3.imshow(x_im,cmap='gray')
plt.show()

