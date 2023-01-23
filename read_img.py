import numpy as np
# import scipy.misc

def read_img(filepath, X=200, Y=200, Z=1024):
    '''

    To read Cirrus scans

    X: number of pixels along X dimension
    Y: number of pixels along Y dimension (number of frames)
    Z: number of pixels along Z dimension

    Output Shape: (1024, 200, 200); Z*X*Y
    '''
    file = open(filepath, 'r')
    dtype = np.dtype('uint8')
    data = np.fromfile(file, dtype)   # (40960000,)
    
    img_vol = np.zeros((Z, X, Y))
    index = 0
    for numFrame in range(Y):
        frameBuffer = data[numFrame*X*Z:(numFrame+1)*X*Z]
#         index = 0
#         frame = np.zeros([X*Z,])
#         for x in range(X):
#             for z in range(Z):
#                 frame[index] = frameBuffer[(Z-1-z)*X + (X-1) - x]
#                 index = index + 1
#         frame = frame.reshape((X, Z))
#         img_vol[:,:,numFrame] = frame.transpose()
#         scipy.misc.imsave(('./sample/%s.tiff' % numFrame), array.transpose())
        frame = frameBuffer.reshape((1024, 200))
        img_vol[:,:,numFrame] = np.fliplr(np.flipud(frame))

    return img_vol