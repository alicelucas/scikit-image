import skimage
from skimage.filters import gaussian
from skimage.filters import motion
from skimage.io import imsave
from skimage import img_as_ubyte

if __name__ == '__main__':
    image = skimage.data.coffee()
    motion_image = motion(image, size = 20, angle = 20.0)
    imsave('~/scikit-image/motion_blurred.png', img_as_ubyte(motion_image))