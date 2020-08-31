import sys
sys.path.append("../")
from utils import *
sys.path.append("../Google_HDRplus/")
from merge_utils import merge
from pwc_on_raw import warped_raw
import cv2

if __name__ == "__main__":
    bk_lvl = 50
    print("black level of the sensor: ", bk_lvl)    
    rIm=read_raw("../static_videos/lowlight3_frame16_bunny_3.raw", frame =16)
    rIm = rIm[:, :2048, :3584]

    for ref_idx in range(0, 16):
        print("merge frames to {}th raw frame...".format(ref_idx))
        ref = rIm[ref_idx]
        final = ref.copy().astype(np.float64)
        for idx in range(0, 16):
            if idx == ref_idx:
                continue
            alt = warped_raw(ref, rIm[idx], display = False)
            H, W = ref.shape
            bi_align_field = np.zeros((H//16-1, W//16-1, 2), dtype=ref.dtype)
            merged_frame = merge(ref, alt, bi_align_field)
            final += merged_frame
    
        print("visualizing and saving...")
        rIm_merge = final[32:-32, 32:-32] - bk_lvl* rIm.shape[0]
        rIm_merge = np.clip(rIm_merge, 0, 1023)
        rIm_merge = rIm_merge.astype(np.uint16)
        cv2.imwrite("rIm_merge_ref{}.png".format(ref_idx), rIm_merge)
        rgbIm_merge = demosaic(rIm_merge)
        rgbIm_adj_merge = adjustColor(rgbIm_merge,rc = 1, bc = 1, gc = 0.7, gain=1.3, gamma = 1.25, contrast=1.5)
        cv2.imwrite("rgbIm_adj_merge_ref{}.png".format(ref_idx), rgbIm_adj_merge[..., ::-1])
        rIm_single = rIm[ref_idx] - bk_lvl
        rIm_single = np.clip(rIm_single, 0, 1023)
        rIm_single = rIm_single.astype(np.uint16)
        cv2.imwrite("rIm_single_ref{}.png".format(ref_idx), rIm_single)
        rgbIm_single = demosaic(rIm_single)
        rgbIm_adj_single = adjustColor(rgbIm_single,rc = 1, bc = 1, gc = 0.7, gain=1.3, gamma = 1.25, contrast=1.5)
        cv2.imwrite("rgbIm_adj_single_ref{}.png".format(ref_idx), rgbIm_adj_single[32:-32, 32:-32, ::-1])
