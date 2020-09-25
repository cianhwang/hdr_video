import numpy as np
import skimage.measure
import cv2

def downsample2x2(rawImg, block_size = 2):
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce
    grayImg = skimage.measure.block_reduce(rawImg, (block_size,block_size), np.mean)
    
    return grayImg

def gauss_pyramid(grayImg, level = 4, scale_list = [2, 2, 2]):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
    assert len(scale_list) + 1 == level
    G = grayImg.copy()
    gpA = [G]
    for i in range(level-1):
        for j in range(scale_list[i]//2):
            G = cv2.pyrDown(G)
        gpA.append(G)
    return gpA

def align_one_level(ref, alt, upsampled_align_field = None, tile_size = 32, radius = 64, norm = 1):

    H, W = ref.shape
    assert ref.shape == alt.shape
    assert H%tile_size == 0 and W%tile_size == 0


    if upsampled_align_field is None:
        upsampled_align_field = np.zeros((H//tile_size, W//tile_size, 2), np.int16)
    align_field = np.zeros_like(upsampled_align_field, upsampled_align_field.dtype)
    
    HH, WW, _ = upsampled_align_field.shape
    assert H//HH == tile_size and W//WW == tile_size
    
    ## non overlap
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            T = ref[i:i+tile_size, j:j+tile_size]
            temp = 1e10

            for u in range(-radius, radius):
                for v in range(-radius, radius):
                    offset_u = u + upsampled_align_field[i//tile_size, j//tile_size, 0]
                    offset_v = v + upsampled_align_field[i//tile_size, j//tile_size, 1]
                    xs = i + offset_u
                    ys = j + offset_v
                    xe = xs + tile_size
                    ye = ys + tile_size

                    if xe > H or ye > W or xs < 0 or ys < 0:
                        continue

                    I_tile = alt[xs:xe, ys:ye]
                    norm_diff = np.linalg.norm(T.flatten()-I_tile.flatten(), ord=norm)
                    if norm_diff < temp:
                        temp = norm_diff
                        align_field[i//tile_size, j//tile_size, 0] = offset_u
                        align_field[i//tile_size, j//tile_size, 1] = offset_v
                        
    return align_field

def upsample_align_field(align_field, scale = 2):
    #upsampled_align_field = 2 * align_field.repeat(2, axis = 0).repeat(2, axis = 1)
    H, W, _ = align_field.shape
    scale = int(scale)
    ## switch H, W here for cv2.resize
    upsampled_align_field = scale * cv2.resize(align_field, (scale * W, scale * H), interpolation = cv2.INTER_NEAREST)
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

def align_gauss_pyramid_hdrplus(gpref, gpalt):
    
    ## gaussian pyramid starts from gray_img: H -> H/2 -> H/8 -> H/32

    assert len(gpref)== len(gpalt)
    
    upsampled_align_field = None
    
    ref, alt = gpref[-1], gpalt[-1]
    align_field = align_one_level(ref, alt, upsampled_align_field, 8, 4, 2)
    upsampled_align_field = upsample_align_field(align_field, 2)  ## scale = scale * tile_size / nex_tile_size
    
    ref, alt = gpref[-2], gpalt[-2]
    align_field = align_one_level(ref, alt, upsampled_align_field, 16, 4, 2)
    upsampled_align_field = upsample_align_field(align_field, 4)
    
    ref, alt = gpref[-3], gpalt[-3]
    align_field = align_one_level(ref, alt, upsampled_align_field, 16, 4, 2)
    upsampled_align_field = upsample_align_field(align_field, 2)

    final_align_field = align_one_level(gpref[0], gpalt[0], upsampled_align_field, 16, 1, 1) ## no rescale needed
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
            if xe > H or ye > W or xs < 0 or ys < 0:
                print(xs, xe, ys, ye)
            alignedImg[i:i+tile_size, j:j+tile_size] = alt[xs:xe, ys:ye]

    return alignedImg
