import sys
sys.path.append("../")

from utils import *
from align_utils import *

rIm=read_raw("../static_videos/lowlight3_frame47_tremor_rg10_2.raw",frame = 16, rows=1087*2, runL=3968)

for idx in range(1, 16):
    ref = rIm[0][:2048, :3584]
    alt  = rIm[idx][:2048, :3584]
    ref, alt = downsample2x2(ref), downsample2x2(alt)
    gpref, gpalt = gauss_pyramid(ref), gauss_pyramid(alt)
    final_align_field = align_gauss_pyramid(gpref, gpalt)
    np.save("final_align_field_{}.npy".format(idx), final_align_field)
    
