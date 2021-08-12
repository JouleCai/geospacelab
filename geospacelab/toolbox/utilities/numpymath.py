import numpy as np

def trig_arctan_to_sph_lon(x, y):
    if np.isscalar(x) and np.isscalar(y):
        isscalar = True
        x = np.array([x])
        y = np.array([y])
    old_settings = np.seterr(divide='ignore')     
    angle = np.arctan(np.divide(y, x))
    angle = np.where(x<0, angle + np.pi, angle)
    angle = np.mod(angle, 2*np.pi)
    np.seterr(**old_settings)
    if isscalar:
        angle = angle[0]
    return angle

