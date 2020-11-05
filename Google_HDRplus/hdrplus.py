from merge_utils import *
from align_utils import *

def HDRPlus_model(imgs, threshold = 5000.):
    n_seq = imgs.shape[0]
    final = imgs[0].copy().astype(np.float64)
    ref = imgs[0]
    for i in range(1, n_seq):
        alt = imgs[i]
        print("merging {}th alternative frame".format(i))
        gray_ref, gray_alt = downsample2x2(ref), downsample2x2(alt)
        gpref, gpalt = gauss_pyramid(gray_ref, 4, [2, 4, 4]), gauss_pyramid(gray_alt, 4, [2, 4, 4])
        final_align_field = align_gauss_pyramid_hdrplus(gpref, gpalt)
        upsampled_align_field = upsample_align_field(final_align_field, 2)
        bi_align_field = bilateral_upsample(upsampled_align_field)
        merged_frame = merge(ref, alt, bi_align_field, 16, 8, threshold)
        final += merged_frame
    return final
    
    
