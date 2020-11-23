import math
import numpy as np

class DataLoader:
    def __init__(self, X:np.array, y:np.array, batch_size:int):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.size, self.length = np.shape(X)
        self.current_batch = 0
    
    def __len__(self):
        return math.floor(self.size / self.batch_size)
    
    def __getitem__(self, key):
        if isinstance( key, slice ) :
        #Get the start, stop, and step from the slice
            return [self[ii] for ii in xrange(*key.indices(len(self)))]
        elif isinstance( key, int ) :
            if key < 0 : #Handle negative indices
                key += len( self )
            if key < 0 or key >= len( self ) :
                raise IndexError('list index out of range')
            
            # get key th batch
            index_begin = self.batch_size * key
            index_end = (self.batch_size ) * (key + 1)
            return self.X[index_begin:index_end], self.y[index_begin:index_end]
        else:
            raise TypeError("Invalid argument type.") 