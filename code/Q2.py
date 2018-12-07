import math
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


def neighbours(i,j,M,N,size=8):
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


def prob_gibbs(target, neighbour_values, observed_value):
    index_nei = 4/len(neighbour_values)
    sigma = 0.5
    nei_similarity = sum(target * neighbour_values)
    probx_givenx = 1 / (1 + math.exp(-nei_similarity * index_nei)) #sigmoid #probx = 1 / (1 + math.exp(-nei_energy)) * math.pow(0.5, len(neighbour_values))
    probmx_givenx = 1 - probx_givenx
    proby_givenx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value - (target+1)/2),2) / (2 * math.pow(sigma,2)))
    proby_givenmx = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value - (target-1)/-2),2) / (2 * math.pow(sigma,2)))
    result = (proby_givenx * probx_givenx)/(proby_givenx * probx_givenx + proby_givenmx * probmx_givenx)
    # print(target,nei_similarity,observed_value,result)
    return result

def gibbs(y):
    x = init(y)
    rows = len(y)
    cols = len(y[0])  # assume not empty
    PASS = 3
    for i in range(PASS):
        for m in range(rows):
            for n in range(cols):
                nei = [x[i] for i in neighbours(m, n, rows, cols)]
                prob = prob_gibbs(1, nei, y[m][n])
                t = np.random.rand()
                if t < prob:
                    x[m][n] = 1
                else:
                    x[m][n] = -1
    return x

def gibbs_rand(y):
    x = init(y)
    rows = len(y)
    cols = len(y[0])  # assume not empty
    PASS = 4*rows*cols
    np.random.seed(42)
    for i in range(PASS):
        m = np.random.randint(rows)
        n = np.random.randint(cols)
        nei = [x[i] for i in neighbours(m, n, rows, cols)]
        prob = prob_gibbs(1, nei, y[m][n])
        t = np.random.rand()
        if t < prob:
            x[m][n] = 1
        else:
            x[m][n] = -1
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
varSigma = 0.5
im = imread('../pic/loli_grey.png')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(141)
ax.imshow(im,cmap='gray')
im_noise = add_gaussian_noise(im,prop,varSigma)
#im_noise = add_saltnpeppar_noise(im,prop)
ax2 = fig.add_subplot(142)
ax2.imshow(im_noise,cmap='gray')
x_im = gibbs(im_noise)
x_im2 = gibbs_rand(im_noise)
ax3 = fig.add_subplot(143)
ax3.imshow(x_im,cmap='gray')
ax4 = fig.add_subplot(144)
ax4.imshow(x_im2,cmap='gray')
plt.show()

