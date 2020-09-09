import sys
sys.path.append("../")
from utils import *
sys.path.append("../Google_HDRplus/")
from merge_utils import merge
sys.path.append("../flow_test/")
from pwc_on_raw import warped_raw
import cv2
from merge_layers import *

if __name__ == "__main__":
    bk_lvl = 50
    print("black level of the sensor: ", bk_lvl)    
    rIm=read_raw("../static_videos/lowlight3_frame16_bunny_3.raw", frame =16)
    rIm = rIm[:, :2048, :3584]
    model = MergeLayerC()
    model.load_state_dict(torch.load('model.pth'))
    model = model.cuda()
    model.eval()

    for ref_idx in range(0, 16):
        print("merge frames to {}th raw frame...".format(ref_idx))
        ref = rIm[ref_idx]
        final = ref.copy().astype(np.float64)
        ref_stack = raw_to_stack(ref)
        ref_t = torch.from_numpy(ref_stack.copy().transpose(2, 0, 1)).float()
        for idx in range(0, 16):
            if idx == ref_idx:
                continue
            flow, _ = warped_raw(ref, rIm[idx], display = False)
            alt = rIm[idx]
            alt_stack = raw_to_stack(alt)
            alt_t = torch.from_numpy(alt_stack.copy().transpose(2, 0, 1)).float()
            flow_t = torch.from_numpy(flow.copy().transpose(2, 0, 1))
            inputs = torch.cat([ref_t, alt_t, flow_t], dim = 0).unsqueeze(0)
            outputs = model(inputs.float().cuda())
            merged_frame = outputs[0].detach().cpu().numpy().transpose(1, 2, 0)
            merged_frame = stack_to_raw(merged_frame)
            final += merged_frame
    
        print("visualizing and saving...")
        rIm_merge = final - bk_lvl* rIm.shape[0]
        rIm_merge = np.clip(rIm_merge, 0, 1023)
        rIm_merge = rIm_merge.astype(np.uint16)
        cv2.imwrite("rIm_merge_ref{}.png".format(ref_idx), rIm_merge)
        rgbIm_merge = demosaic(rIm_merge)
        rgbIm_adj_merge = adjustColor(rgbIm_merge,rc = 1, bc = 1, gc = 0.7, gain=1.3, gamma = 1.25, contrast=1.5)
        cv2.imwrite("rgbIm_adj_merge_ref{}.png".format(ref_idx), rgbIm_adj_merge[..., ::-1])
