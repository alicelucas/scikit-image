from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import _ni_support, correlate1d, convolve
from ..transform import rotate
from ..util import img_as_float
from .._shared.utils import warn, convert_to_float

__all__ = ['motion']

def motion(image, size=5, angle=30, output=None, preserve_range=False):
    """Motion filter for RGB images.

    Parameters 
    -----------
    image : array-like
        Input image(grayscale or color) to filter.
    size : scalar, optional
        Size of the motion blur. 
    angle : float, optional
        Angle (direction) of the motion blur.
    output : array, optional
        The ``output`` parameter passes an array in which to store 
        the filter output.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of ``img_as_float``.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    """
  
    if not image.ndim == 3 and not image.ndim == 2:
        raise ValueError("Expected 2D or 3D array, got %iD." % image.ndim)
    if image.ndim == 3:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB "
               "by default. Filtering will be performed independently on each channel.")
        warn(RuntimeWarning(msg))
    if size < 0:
        raise ValueError("Motion blur size values less than zero are not valid")
    image = convert_to_float(image, preserve_range)
    if output is None:
        output = np.empty_like(image)
    elif not np.issubdtype(output.dtype, np.floating):
        raise ValueError("Provided output data type is not float")

    image = np.asarray(image)
    output = _ni_support._get_output(output, image)
    axes = list(range(3)) if image.ndim == 3 else []
    weights = motion_kernel(size, angle)
    if len(axes) > 0:
        for axis in axes:
            convolve(image[:,:,axis], weights, output[:,:,axis])
    else:
        convolve(image, weights, output)
    return output


def motion_kernel(size, angle):
    """ Kernel for motion blur. 
    Defined as SIZE x SIZE matrix to be convolved over the spatial axes of the image.

    Parameters
    -----------
    size : scalar, optional
        Size of the motion blur. 
    angle : float, optional
        Angle (direction) of the motion blur.
    """
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    k = rotate(k, angle)
    k = k / np.sum(k)
    return k