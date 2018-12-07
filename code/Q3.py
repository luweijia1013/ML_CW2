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

def energy(target, neighbour_values, observed_value):
    bias = 0
    index_nei = 1
    index_xy = 8
    #(0,1,2) for tmac2, loop5 (0.7,1)
    #(0,1,2) for pug, loop6 (0.7,1)
    energy = bias * target - index_nei * target * sum(neighbour_values) - index_xy * target * (observed_value - 0.5 )
    return energy

def prob(target, neighbour_values, observed_value):
    return -energy(target, neighbour_values, observed_value)#simplify the prob for convinience


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

def icm(y):
    x = init(y)
    rows = len(y)
    cols = len(y[0]) #assume not empty
    PASS = 20
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

def likelihood(target, observed_value):
    sigma = 1
    return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow((observed_value - (target+1)/2),2) / (2 * math.pow(sigma,2)))

def q(target, para_m_value, observed_value):
    inside = 2 * (para_m_value + 0.5 * (likelihood(1, observed_value) - likelihood(-1, observed_value) ) )
    # print (inside)
    return 1 / (1 + np.exp(-inside))

def variational_bayes(y):
    x = init(y)
    para_m = np.copy(y)
    para_miu = np.copy(y)
    para_w = 0.75
    miu_init = 0
    rows = len(y)
    cols = len(y[0])  # assume not empty
    PASS = 20
    for m in range(rows):
        for n in range(cols):
            para_m[(m,n)] = 0
            para_miu[(m,n)] = miu_init
    # print (para_m,para_miu)
    for i in range(PASS):
        for m in range(rows):
            for n in range(cols):
                # if m == 80 and n == 80:
                    # print(para_m[(m,n)],para_miu[(m,n)],i)
                nei_len = len(neighbours(m,n,rows,cols))
                para_m[(m,n)] = para_w/nei_len * sum(para_miu[index] for index in neighbours(m,n,rows,cols))
                para_miu[(m,n)] = np.tanh(para_m[(m,n)] + 0.5 * (likelihood(1, y[(m,n)]) - likelihood(-1, y[(m,n)]) ) )
    for m in range(rows):
        for n in range(cols):
            judge = q(1, para_m[(m,n)], y[(m,n)])
            if judge > 0.5:
                x[(m,n)] = 1
            else:
                x[(m,n)] = -1
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
im = imread('../pic/char_grey.png')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(151)
ax.imshow(im,cmap='gray')
im_noise = add_gaussian_noise(im,prop,varSigma)
#im_noise = add_saltnpeppar_noise(im,prop)
ax2 = fig.add_subplot(152)
ax2.imshow(im_noise,cmap='gray')
im_icm = icm(im_noise)
ax3 = fig.add_subplot(153)
ax3.imshow(im_icm,cmap='gray')
im_gibbs = gibbs(im_noise)
# im_gibbs = gibbs_rand(im_noise)
ax4 = fig.add_subplot(154)
ax4.imshow(im_gibbs,cmap='gray')
im_vb = variational_bayes(im_noise)
ax5 = fig.add_subplot(155)
ax5.imshow(im_vb,cmap='gray')

plt.show()

