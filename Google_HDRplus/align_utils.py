import numpy as np
import skimage.measure
import cv2

def downsample2x2(rawImg, block_size = 2):
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce
    grayImg = skimage.measure.block_reduce(rawImg, (block_size,block_size), np.mean)
    
    return grayImg

def gauss_pyramid(grayImg, level = 4):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
    G = grayImg.copy()
    gpA = [G]
    for i in range(level-1):
        G = cv2.pyrDown(G)
        gpA.append(G)
    return gpA

def align_one_level(ref, alt, upsampled_align_field = None, tile_size = 32):

    H, W = ref.shape
    assert ref.shape == alt.shape
    assert H%tile_size == 0
    assert W%tile_size == 0

    if upsampled_align_field is None:
        upsampled_align_field = np.zeros((H//tile_size, W//tile_size, 2), np.int16)
    align_field = np.zeros_like(upsampled_align_field, upsampled_align_field.dtype)
    
    ## non overlap
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            T = ref[i:i+tile_size, j:j+tile_size]
            temp = 1e10

            for u in range(-64, 64):
                for v in range(-64, 64):
                    offset_u = u + upsampled_align_field[i//tile_size, j//tile_size, 0]
                    offset_v = v + upsampled_align_field[i//tile_size, j//tile_size, 1]
                    xs = i + offset_u
                    ys = j + offset_v
                    xe = xs + tile_size
                    ye = ys + tile_size

                    if xe > H or ye > W or xs < 0 or ys < 0:
                        continue

                    I_tile = alt[xs:xe, ys:ye]
                    l1_diff = np.linalg.norm(T.flatten()-I_tile.flatten(), ord=1)
                    if l1_diff < temp:
                        temp = l1_diff
                        align_field[i//tile_size, j//tile_size, 0] = offset_u
                        align_field[i//tile_size, j//tile_size, 1] = offset_v
                        
    return align_field

def upsample_align_field(align_field):
    upsampled_align_field = 2 * align_field.repeat(2, axis = 0).repeat(2, axis = 1)
    return upsampled_align_field

def align_gauss_pyramid(gpref, gpalt, tile_size = 32):

    assert len(gpref)== len(gpalt)
    upsampled_align_field = None
    print("align from coarse to fine...")
    idx = 1
    for ref, alt in zip(gpref[len(gpref):0:-1], gpalt[len(gpalt):0:-1]):
        print("level", idx, "aligning")
        align_field = align_one_level(ref, alt, upsampled_align_field, tile_size)
        upsampled_align_field = upsample_align_field(align_field)
        idx += 1
    print("level", idx, "aligning")
    final_align_field = align_one_level(gpref[0], gpalt[0], upsampled_align_field)
    return final_align_field

def align_final(alt, final_align_field, tile_size = 32):
    alignedImg = np.zeros_like(alt, alt.dtype)
    H, W = alt.shape
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            xs = i + final_align_field[i//tile_size, j//tile_size, 0]
            ys = j + final_align_field[i//tile_size, j//tile_size, 1]
            xe = xs + tile_size
            ye = ys + tile_size
            alignedImg[i:i+tile_size, j:j+tile_size] = alt[xs:xe, ys:ye]

    return alignedImg
