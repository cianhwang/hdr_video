import sys
sys.path.append("../")
import os.path

from utils import *
from align_utils import *

rIm=read_raw("../static_videos/lowlight3_frame16_bunny.raw",frame = 16, rows=1087*2, runL=3968)

for ref_idx in range(0, 16):
    for idx in range(0, 16):
        ### check file existance
        if os.path.exists('lowlight3_bunny/final_align_field_ref{}_{}.npy'.format(ref_idx, idx)):
            print('lowlight3_bunny/final_align_field_ref{}_{}.npy exists'.format(ref_idx, idx))
            continue
        if ref_idx == idx:
            continue
        print("ref:", ref_idx, "current:", idx, "th frame")
        ref = rIm[ref_idx][:2048, :3584]
        alt  = rIm[idx][:2048, :3584]
        ref, alt = downsample2x2(ref), downsample2x2(alt)
        gpref, gpalt = gauss_pyramid(ref), gauss_pyramid(alt)
        final_align_field = align_gauss_pyramid(gpref, gpalt)
        np.save("lowlight3_bunny/final_align_field_ref{}_{}.npy".format(ref_idx, idx), final_align_field)
    
