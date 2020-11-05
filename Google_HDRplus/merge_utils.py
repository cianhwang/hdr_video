import numpy as np
import cv2

def merge_distance(T, I_tile, threshold = 1420.4/4): #1037.916
    assert T.shape == I_tile.shape
    weight = 0
    dist = np.linalg.norm(T.flatten()-I_tile.flatten(), ord=1)
#     print(dist)
    if dist < threshold:
        weight = 1
    return weight

def raised_cosine_filter(tile_size = 32):
    x = np.arange(tile_size)
    xv, yv = np.meshgrid(x, x)
    window = (1-np.cos(2*np.pi*(xv+0.5)/tile_size))*(1-np.cos(2*np.pi*(yv+0.5)/tile_size))/4
    return window

def bilateral_upsample(align_field):
    ## H x W --> (2H-1) x (2W-1)
    #bi_align_field = align_field.repeat(2, axis=0).repeat(2, axis=1)
    H, W, _ = align_field.shape
    bi_align_field = cv2.resize(align_field, (2 * W, 2 * H), interpolation = cv2.INTER_NEAREST)
    return bi_align_field[:-1, :-1]

def merge(ref, alt, bi_align_field, tile_size = 32, stride = 16, threshold = 1420.4/4):
    ## ref and alt are raw Bayer data.
    assert ref.shape == alt.shape
    H, W = ref.shape
    merged_frame = np.zeros_like(ref, dtype = np.float64) # existance of cosine filter, dtype are different.
    for i in range(0, H - stride, stride):
        for j in range(0, W - stride, stride):
            T = ref[i:i+tile_size, j:j+tile_size]
            xs = i + bi_align_field[i//stride, j//stride, 0]
            ys = j + bi_align_field[i//stride, j//stride, 1]
            xe = xs + tile_size
            ye = ys + tile_size
            if xs < 0 or ys < 0 or xe > H or ye > W: ## only process lie-in-between tiles
                print("error: xs, ys, xe, ye", xs, ys, xe, ye)
                print("error: i, j, i%32, j%32", i, j, i%32, j%32)
                I_tile = T.copy()
            else:
                I_tile = alt[xs:xe, ys:ye]
            weight = merge_distance(T, I_tile, threshold)
            merged_tile = weight * I_tile + (1-weight) * T
            merged_frame[i:i+tile_size, j:j+tile_size] += merged_tile.astype(np.float64) * raised_cosine_filter(tile_size)

    return merged_frame

if __name__ == '__main__':
    
    print("test merge dist...")
    
    I = np.random.rand(32, 32)
    T = np.random.rand(32, 32)
    print("weight:", merge_distance(T, I, 340.0))
    
    print("test raised_cosine_filter ...")
    print(raised_cosine_filter())
    
    print("test bilateral_upsample ...")
    align_field = np.random.randint(256, size=(64, 112))
    print(align_field[:2, :4])
    print(bilateral_upsample(align_field).shape, bilateral_upsample(align_field)[:4,:8])
    
    


            
            
