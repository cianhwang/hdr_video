import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt

def print_stat(narray_name, narray):
    print(narray_name, "shape: ", narray.shape, "dtype:", narray.dtype)
    arr = narray.flatten()
    print(narray_name , "stat: max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))

def read_raw(fileName, frame = 1, rows = 2174, runL = 3968):

    rIm = np.fromfile(fileName, dtype = np.dtype('i2'))

    rIm = np.reshape(rIm[:frame*rows*runL], (frame, rows, runL))

    return rIm[..., :3864][:, ::-1, ::-1]

def demosaic(rawImage):  
    H, W = rawImage.shape
    red=rawImage[0:H:2,0:W:2]
    redU=interp.RectBivariateSpline(np.arange(0,H//2),np.arange(0,W//2),red)
    red=redU(np.arange(0,H//2,.5),np.arange(0,W//2,.5))
    blue=rawImage[1:H:2,1:W:2]
    blueU=interp.RectBivariateSpline(np.arange(0,H//2),np.arange(0,W//2),blue)
    blue=blueU(np.arange(0,H//2,.5),np.arange(0,W//2,.5))
    green=rawImage;
    for er in range(0,H-1,2):
        for odr in range(0,W-2,2):
            green[er,odr]=np.mean([green[er-1,odr], green[er,odr-1],green[er+1,odr], green[er,odr+1]])
    for er in range(1,H-1,2):
        for odr in range(1,W-2,2):
            green[er,odr]=np.mean([green[er-1,odr], green[er,odr-1],green[er+1,odr], green[er,odr+1]])
    greenU=interp.RectBivariateSpline(np.arange(0,H),np.arange(0,W-2),green[0:H,0:W-2])
    green=greenU(np.arange(0,H,1),np.arange(0,W,1))
    imageOut=np.dstack((red,green,blue))
    imageOut=255*imageOut/np.max(imageOut)
    imageOut=imageOut.astype(np.uint8)
    return imageOut

def adjustColor(inputImage,rc=1,bc=1,gc=1,gain=1,gamma = 0.5, contrast=1):
    fI=inputImage.astype(np.float)
    fI[:,:,0]=rc*fI[:,:,0]
    fI[:,:,1]=gc*fI[:,:,1]
    fI[:,:,2]=bc*fI[:,:,2]#
    fI=gain*fI/np.max(fI)
    fI = np.power(fI, 1/gamma)
    fI=255*np.tanh(contrast*fI)
    fI=fI.astype(np.uint8)
    return fI

def plot_histogram(Im, bit_length = 8):
    if len(Im.shape) == 3: # rgb image
        red, green, blue = Im[..., 0], Im[..., 1], Im[..., 2]
    elif len(Im.shape) == 2: # raw image RGGB
        red, g1, g2, blue = Im[::2, ::2], Im[1::2, ::2], Im[::2, 1::2], Im[1::2, 1::2]
        green = [g1, g2]
    else:
        raise Exception("wrong dimension.")
        
    max_value = 2**bit_length
    
    histr,bins=np.histogram(red,np.arange(0,max_value))
    histg,bins=np.histogram(green,np.arange(0,max_value))
    histb,bins=np.histogram(blue,np.arange(0,max_value))
    
    plt.plot(np.arange(0, max_value-1),histr,'r', np.arange(0,max_value-1),histg,'g', np.arange(0,max_value-1),histb, 'b')
    plt.show()
    
def plot_histogram_normalized(Im, max_range = 1.):
    if len(Im.shape) == 3: # rgb image
        red, green, blue = Im[..., 0], Im[..., 1], Im[..., 2]
    elif len(Im.shape) == 2: # raw image RGGB
        red, g1, g2, blue = Im[::2, ::2], Im[1::2, ::2], Im[::2, 1::2], Im[1::2, 1::2]
        green = [g1, g2]
    else:
        raise Exception("wrong dimension.")
        
    max_value = 1.
    
    histr,bins=np.histogram(red,np.linspace(0,max_value))
    histg,bins=np.histogram(green,np.linspace(0,max_value))
    histb,bins=np.histogram(blue,np.linspace(0,max_value))
    
    plt.plot(np.linspace(0, max_range, 49, endpoint=False), histr,'r', 
             np.linspace(0, max_range, 49, endpoint=False), histg,'g', 
             np.linspace(0, max_range, 49, endpoint=False), histb, 'b')
    plt.show()
    
def raw_to_stack(raw_img, pattern='rggb'):
    """Reshape the raw image into depth 4 stack, following order rggb, depth on last channel"""
    A,B,C,D = raw_img[:-1:2, :-1:2], \
              raw_img[:-1:2, 1: :2], \
              raw_img[1: :2, :-1:2], \
              raw_img[1: :2, 1: :2]
    if pattern.lower() == 'rggb':
        return np.stack([A,B,C,D], axis=-1)
    else:
        raise NotImplementedError

def stack_to_raw(rggb_stack):
    H, W, _ = rggb_stack.shape
    raw_img = np.zeros((2*H, 2*W), rggb_stack.dtype)
    raw_img[:-1:2, :-1:2] = rggb_stack[..., 0]
    raw_img[:-1:2, 1: :2] = rggb_stack[..., 1]
    raw_img[1: :2, :-1:2] = rggb_stack[..., 2]
    raw_img[1: :2, 1: :2] = rggb_stack[..., 3]
    return raw_img

def viz_raw(raw_image):
    H, W = raw_image.shape
    viz_raw_img = np.zeros((H, W, 3), raw_image.dtype)
    viz_raw_img[:-1:2, :-1:2, 0] = raw_image[:-1:2, :-1:2]
    viz_raw_img[:-1:2, 1: :2, 1] = raw_image[:-1:2, 1: :2]
    viz_raw_img[1: :2, :-1:2, 1] = raw_image[1: :2, :-1:2]
    viz_raw_img[1: :2, 1: :2, 2] = raw_image[1: :2, 1: :2]
    return viz_raw_img

def viz_stack(stack_raw):
    H, W, C = stack_raw.shape
    assert C == 4
    r = np.zeros((H, W, 3), stack_raw.dtype)
    g1 = np.zeros((H, W, 3), stack_raw.dtype)
    g2 = np.zeros((H, W, 3), stack_raw.dtype)
    b = np.zeros((H, W, 3), stack_raw.dtype)
    
    r[..., 0] = stack_raw[..., 0]
    g1[..., 1] = stack_raw[..., 1]
    g2[..., 1] = stack_raw[..., 2]
    b[..., 2] = stack_raw[..., 3]
    return r, g1, g2, b
    

def print_model_params(model):
    print("#total params:", sum(p.numel() for p in model.parameters()), end='')
    print(" | #trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
def add_window(img, left_x, right_x, color = 'r', win_size = 256):
    assert img.dtype == np.uint8
    img_w_win = img.copy().astype(np.int16)
    patch = np.zeros((win_size, win_size, 3), dtype = np.int16)
    if color == 'r':
        patch[..., 0] = 255
    elif color == 'g':
        patch[..., 1] = 255
    elif color == 'b':
        patch[..., 2] = 255
    elif color == 'y':
        patch[..., [0, 1]] = 255
    elif color == 'p':
        patch[..., [0, 2]] = 255
    elif color == 'c':
        patch[..., [1, 2]] = 255
    else:
        raise NotImplementedError('unrecognized color type')
    patch[10:-10, 10:-10, :] = 0
    
    img_w_win[left_x:left_x+win_size, right_x:right_x+win_size] += patch
    img_w_win = np.clip(img_w_win, 0, 255)
    return img_w_win.astype(np.uint8)

def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 10 * np.log10(PIXEL_MAX**2 / mse)

