import math
import numpy as np
#
# DATA TYPES
#

class PointBase(np.ndarray):
    ''' Base class for points. 
        
        Provide access to ndarray elements via named fields.
        ndarray have the same size as number of fields.
    '''
    __fields_names__ = ()

    def __new__(cls, *args):
        # check data shape
        if not all(np.isreal(arg) and np.isscalar(arg) for arg in args):
            raise TypeError(f'invalid arguments: expected {len(cls.__fields_names__)} real numbers')
        # parse initial values
        ndim = len(cls.__fields_names__)
        if len(args) == ndim:
            buffer = np.array(args, dtype=np.float64)
        elif len(args) == 0:
            buffer = np.zeros(shape=(ndim,), dtype=np.float64)
        else:
            raise TypeError(f'invalid arguments: expected 0 or {ndim} arguments')
        # ndarray construction
        return super(PointBase, cls).__new__(cls, shape=(ndim,), dtype=np.float64, buffer=buffer)

    def __init__(self, *args):
        super(PointBase, self).__init__()

    def __array_wrap__(self, arr, context=None):
        if arr.shape == self.shape:
            return super(PointBase, self).__array_wrap__(arr, context)
        else:
            return arr

    def __init_subclass__(cls, fields, **kwargs):
        ''' Called upon subclass creation. It adds fields getter and setters. '''
        super(PointBase, cls).__init_subclass__(**kwargs)
        # store fields list
        cls.__fields_names__ = fields
        # add getter and setters for fields
        def get_getter(k):
            return lambda self: self[k]
        def get_setter(k):
            return lambda self, value: self.__setitem__(k, value)
        for k, field in enumerate(cls.__fields_names__):
            setattr(cls, field, property(get_getter(k), get_setter(k)))

    def __getitem__(self, key):
        ''' Access to elements by fields names. '''
        if key in self.__fields_names__:
            return getattr(self, key)
        else: 
            return super(PointBase, self).__getitem__(key)
        
    def __setitem__(self, key, value):
        if key in self.__fields_names__:
            setattr(self, key, value)
        else:
            return super(PointBase, self).__setitem__(key, value)

    def norm(self):
        return np.sqrt(np.sum(self ** 2))
            
    def __repr__(self):
        return f"{type(self).__name__} {np.array(self.data)}"
    
    def __str__(self):
        return np.array(self).__str__()

class PointDQ(PointBase, fields = ('d', 'q')):
    def __init__(self, d = 0.0, q = 0.0):
        super(PointDQ, self).__init__(d, q)

    def park_inv(self, EAngle):
        return PointXY(*park_inv(*self, EAngle))

class PointXY(PointBase, fields = ('x', 'y')):
    def __init__(self, x = 0.0, y = 0.0):
        super(PointXY, self).__init__(x, y)

    def clark_inv(self):
        return PointABC(*clark_inv(*self))

    def park(self, EAngle):
        return PointDQ(*park(*self, EAngle))

class PointABC(PointBase, fields = ('a', 'b', 'c')):
    def __init__(self, a = 0.0, b = 0.0, c = 0.0):
        super(PointABC, self).__init__(a, b, c)

    def clark(self):
        return PointXY(*clark(*self))
    
#
# HELPER FUNCTIONS
#
    
def clark_inv(Ux, Uy):
    ''' Inverse Clark transform: convert two-phase reperesentation to tree phases.

    Parameters
    ----------
    Ux: array_like or numeric
    Uy: array_like or numeric

    Returns
    -------
    Ua: array_like or numeric
    Ub: array_like or numeric
    Uc: array_like or numeric
    '''
    Ua = Ux
    Ub = -0.5*Ux + (math.sqrt(3.0)/2.0)*Uy
    Uc = -0.5*Ux - (math.sqrt(3.0)/2.0)*Uy
    return Ua, Ub, Uc
    
def clark(*args):
    ''' Clark transform: convert three phase reperesentation to two phases.

    Parameters
    ----------
    Ua: array_like or numeric
    Ub: array_like or numeric
    Uc: array_like or numeric, optional

    Returns
    -------
    Ux: array_like or numeric
    Uy: array_like or numeric
    '''
    if len(args) == 2:
        Ua, Ub = args
        Ux = Ua
        Uy = (Ua + 2*Ub)/math.sqrt(3)
    elif len(args) == 3:
        Ua, Ub, Uc = args
        Ux = (2.0/3.0)*Ua - (1.0/3.0)*Ub - (1.0/3.0)*Uc
        Uy = (math.sqrt(3.0)/3.0)*(Ub - Uc)
    else:
        raise ValueError('clark() accepts two or three arguments')
    return Ux, Uy

def park_inv(Id, Iq, EAngle):
    ''' Inverse Park transform: convert rotating frame to stationary.

    Parameters
    ----------
    Id: array_like or numeric
        Rotating frame d-axis.
    Iq: array_like or numeric
        Rotating frame q-axis.
    EAngle: array_like or numeric
        Electrical angle in radians.

    Returns
    -------
    Ix: array_like or numeric
        Rotating frame x-axis (a-axis).
    Iy: array_like or numeric
        Rotating frame y-axis (b-axis).
    '''
    cos = np.cos(EAngle)
    sin = np.sin(EAngle)
    Ix = cos*Id - sin*Iq
    Iy = sin*Id + cos*Iq
    return Ix, Iy

def park(Ix, Iy, EAngle):
    ''' Park transform: convert stationary frame to rotating.

    Parameters
    ----------
    EAngle: array_like or numeric
    Ix: array_like or numeric
        Rotating frame x-axis (a-axis).
    Iy: array_like or numeric
        Rotating frame y-axis (b-axis).
    EAngle: array_like or numeric
        Electrical angle in radians.

    Returns
    -------
    Id: array_like or numeric
        Rotating frame d-axis.
    Iq: array_like or numeric
        Rotating frame q-axis.
    '''
    cos = np.cos(EAngle)
    sin = np.sin(EAngle)
    Id =   cos*Ix + sin*Iy
    Iq = - sin*Ix + cos*Iy
    return Id, Iq
