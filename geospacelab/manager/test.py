
import numpy as np
class Variable(np.ndarray):
    __attrs = ['aa']
    __aa = 0
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)


if __name__ == "__main__":
    a = Variable([1, 2, 3, 4])
    pass