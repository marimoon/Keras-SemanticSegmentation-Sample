import random
import numpy as np

def create_tile(train_x, train_y, num_sqrt=4, avoid_nums=[]):
    shape = train_x[0].shape
    agg_indexs = [ np.where(train_y==val)[0].tolist() for val in range(10) if not val in avoid_nums ]
    ext_indexs = [ random.choice(agg_indexs[i%len(agg_indexs)]) for i in range(num_sqrt*num_sqrt) ]
    img = np.zeros((num_sqrt*shape[0], num_sqrt*shape[1]))
    random.shuffle(ext_indexs)
    for i, ind in enumerate(ext_indexs):
        x_i = i%num_sqrt
        y_i = i//num_sqrt
        #print(x_i, y_i, train_y[ind])
        img[ y_i*shape[0]:(y_i+1)*shape[0], x_i*shape[1]:(x_i+1)*shape[1] ] = train_x[ind]
    return img

def create_batch_tile(train_x, train_y, batch_size=16, num_sqrt=4, avoid_nums=[]):
    base = np.empty((0,)+train_x[0].shape)
    imgs = [ np.array([create_tile(train_x, train_y, num_sqrt=num_sqrt, avoid_nums=avoid_nums)]) for _ in range(batch_size) ]
    base = np.vstack(imgs)
    base = base.reshape((batch_size,)+imgs[0].shape[1:3]+(1,) )
    return base
