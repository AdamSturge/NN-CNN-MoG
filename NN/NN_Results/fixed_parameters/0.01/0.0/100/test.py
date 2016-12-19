import numpy as np
import matplotlib.pyplot as plt
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))
               

model = np.load("nn_model.npz")
W1 = np.reshape(model['W1'],[48,48,7])
W1 = np.transpose(W1,[2,0,1])
W1 = unblockshaped(W1,7*48,1*48)
W1 = np.append(W1,np.zeros([48,48]), axis=0)
W11 = W1[0:4*48,:]
W12 = W1[4*48:,:]

print(W11.shape)
print(W12.shape)

W1 = np.concatenate((W11,W12),axis=1)

plt.matshow(W1,fignum=0,cmap=plt.cm.gray)
plt.show()

#plt.savefig(filename="first_4.png")

#plt.matshow(W12,fignum=1,cmap=plt.cm.gray)
#plt.show()

#plt.savefig(filename="last_3.png")
