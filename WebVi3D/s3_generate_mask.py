import argparse
import glob
import os

import cv2
import numpy as np
import skimage.morphology
import torch
from PIL import Image
from tqdm import tqdm
try:
    from filelock import FileLock
except ImportError:
    os.system("pip install filelock")
    from filelock import FileLock
DEVICE = "cpu"


import threading
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)  # N 3
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)  # N 3
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z**2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def get_stats(X, norm=2):
    """
    :param X (N, C, H, W)
    :returns mean (1, C, 1, 1), scale (1)
    """
    mean = X.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="examples_vid/step1+2_outputs",help="Dataset path")
    parser.add_argument("--txt", type=str, default="examples_txt/mask_list.txt", help="txt file to store processed video names,use it for the following steps.")
    args = parser.parse_args()
    done = []

    mask_done_file = args.txt
    lockpath = mask_done_file + ".lock"
    lock = FileLock(lockpath)
    if os.path.exists(mask_done_file):
        with lock:
            done = set()
            with open(mask_done_file, 'r') as f:
                for line in f:
                    done.add(line.strip().split(' >')[0])
    for data in os.listdir(args.dataset_path):
        if data in done:
            continue
        data_dir = os.path.join(args.dataset_path, data)
        if not os.path.exists(os.path.join(data_dir, 'flow')):
            print("[WARNING] no flow detected, aborted", os.path.join(data_dir, 'flow'))
            with lock:
                with open(mask_done_file, "a") as f:
                    f.write(data+" >>> noflow\n")
                    f.close()
                    continue
        with lock:
            with open(mask_done_file, "a") as f:
                f.write(data+" >>>done\n")
                f.close()

        images = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")), key=extract_number)[:200]

        img = load_image(images[0])
        H = img.shape[2]
        W = img.shape[3]
        uv = get_uv_grid(H, W, align_corners=False)
        x1 = uv.reshape(-1, 2)  # N, 2
        motion_mask_frames = []
        motion_mask_frames2 = []
        flow_for_bilateral = []

        with tqdm(images) as pbar:
            for idx, _ in enumerate(images):
                # print("idx: " + str(idx))
                pbar.set_description("Processing {}".format(idx))
                motion_masks = []
                weights = []
                err_list = []
                normalized_flow = []
                this_flow = 0
                counter = 0
                for step in [1]:
                    # print("step: " + str(step))
                    if idx - step >= 0:
                        # backward flow and mask
                        bwd_flow_path = os.path.join(
                            data_dir, "flow", str(idx).zfill(5) + "_bwd.npz"
                        )
                        if not os.path.exists(bwd_flow_path):
                            print("Flow not preprared, skipping "+ data_dir)
                            continue
                        bwd_data = np.load(bwd_flow_path)
                        bwd_flow, bwd_mask = bwd_data["flow"], bwd_data["mask"]
                        this_flow = np.copy(this_flow - bwd_flow)
                        counter += 1
                        bwd_flow = torch.from_numpy(bwd_flow)
                        bwd_mask = np.float32(bwd_mask)
                        bwd_mask = torch.from_numpy(bwd_mask)
                        flow = torch.from_numpy(
                            np.stack(
                                [
                                    2.0 * bwd_flow[..., 0] / (W - 1),  # normalized to -1,1
                                    2.0 * bwd_flow[..., 1] / (H - 1),
                                ],
                                axis=-1,
                            )
                        )
                        normalized_flow.append(flow)
                        x2 = x1 + flow.view(-1, 2)  # (H*W, 2),
                        randid = [np.random.randint(0, x1.shape[0]) for _ in range(500)]
                        F, mask = cv2.findFundamentalMat(x1.numpy()[randid], x2.numpy()[randid], cv2.FM_LMEDS) 
                        F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
                        err = compute_sampson_error(x1, x2, F).reshape(H, W)
                        fac = (H + W) / 2
                        err = err * fac ** 2
                        err_list.append(err)
                        weights.append(bwd_mask.mean())

                    if idx + step < len(images):
                        # forward flow and mask
                        fwd_flow_path = os.path.join(
                            data_dir, "flow", str(idx).zfill(5) + "_fwd.npz"
                        )
                        if not os.path.exists(fwd_flow_path):
                            print("Flow not preprared, skipping "+ data_dir)
                            continue
                        fwd_data = np.load(fwd_flow_path)
                        fwd_flow, fwd_mask = fwd_data["flow"], fwd_data["mask"]
                        this_flow = np.copy(this_flow + fwd_flow)
                        counter += 1
                        fwd_flow = torch.from_numpy(fwd_flow)
                        fwd_mask = np.float32(fwd_mask)
                        fwd_mask = torch.from_numpy(fwd_mask)
                        flow = torch.from_numpy(
                            np.stack(
                                [
                                    2.0 * fwd_flow[..., 0] / (W - 1),
                                    2.0 * fwd_flow[..., 1] / (H - 1),
                                ],
                                axis=-1,
                            )
                        )
                        normalized_flow.append(flow)
                        x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
                        randid = [np.random.randint(0, x1.shape[0]) for _ in range(500)]
                        F, mask = cv2.findFundamentalMat(x1.numpy()[randid], x2.numpy()[randid], cv2.FM_LMEDS)
                        F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
                        err = compute_sampson_error(x1, x2, F).reshape(H, W)
                        fac = (H + W) / 2
                        err = err * fac**2
                        err_list.append(err)
                        weights.append(fwd_mask.mean())
                if len(err_list) > 0:
                    err = torch.amax(torch.stack(err_list, 0), 0)
                    flow_for_bilateral.append(this_flow / counter)

                    thresh = torch.quantile(err, 0.8)   # threshold
                    err = torch.where(err <= thresh, torch.zeros_like(err), err)

                    mask = torch.from_numpy(
                        skimage.morphology.binary_opening(
                            err.numpy() > (H * W / (8100.0)), skimage.morphology.disk(1)
                        )
                    )  
                    mask = torch.from_numpy(
                        skimage.morphology.dilation(mask.cpu(), skimage.morphology.disk(2))
                    ).float()
                    if not os.path.exists(os.path.join(data_dir, "dynamic_mask")):
                        os.makedirs(os.path.join(data_dir, "dynamic_mask"))
                    Image.fromarray(((mask).numpy() * 255.0).astype(np.uint8)).save(
                        os.path.join(data_dir, "dynamic_mask", str(idx).zfill(5) + ".png")
                    )

                    pbar.update()
    print(f"mask done. please see {mask_done_file}")
