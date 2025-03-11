import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from utils.RAFT.raft import RAFT
from utils.RAFT.utils.utils import InputPadder

from utils.flow_utils import *
from tqdm import tqdm
DEVICE = "cuda"
import shutil
try:
    from filelock import FileLock
except ImportError:
    os.system("pip install filelock")
    from filelock import FileLock

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(imfile):
    long_dim = 768
    img = np.array(Image.open(imfile)).astype(np.uint8)

    # Portrait Orientation
    if img.shape[0] > img.shape[1]:
        input_h = long_dim
        input_w = int(round(float(input_h) / img.shape[0] * img.shape[1]))
    # Landscape Orientation
    else:
        input_w = long_dim
        input_h = int(round(float(input_w) / img.shape[1] * img.shape[0]))

    # print("flow input w %d h %d" % (input_w, input_h))
    img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask, bwd_mask

def check_flow(folder_path):
    flow = sorted(glob.glob(os.path.join(folder_path, 'flow') + '/*bwd.npz'))
    count = 0
    for f in flow[:20]:
        npz = np.load(f)
        flow = npz['flow']
        move = abs(flow)[...,0]+ abs(flow)[...,1]
        if move.mean() <= 0.5:
            count += 1
    return count

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def camera_movement(imglist):
    thres = 0
    prev = None
    for img in imglist[::2]:
        img = np.array(Image.open(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        if prev is not None:
            ssim = cv2.matchTemplate(edges, prev, cv2.TM_CCOEFF_NORMED)[0][0]
            # print(ssim)
            if ssim >= 0.7: thres += 1  # changable, higher means more strict
        prev = edges

    return thres

def check_flow(flow):
    # flow = sorted(glob.glob(os.path.join(folder_path, 'flow') + '/*bwd.npz'))
    move = abs(flow)[...,0]+ abs(flow)[...,1]
    if move.mean() <= 0.4:
        return True
    return False


def run(args, input_path, output_path, output_img_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    small_cnt = 0
    cnt = 0

    with torch.no_grad():
        images = glob.glob(os.path.join(input_path, "*.png")) + glob.glob(
            os.path.join(input_path, "*.jpg")
        )

        images = sorted(images)[:200]   # max 200 frames
        img_train = cv2.imread(images[0])
        for i in tqdm(range(len(images) - 1)):
            # print(i)
            image1 = load_image(images[i])
            image2 = load_image(images[i + 1])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_fwd = model(image1, image2, iters=20, test_mode=True)
            _, flow_bwd = model(image2, image1, iters=20, test_mode=True)

            flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
            flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

            flow_fwd = resize_flow(flow_fwd, img_train.shape[0], img_train.shape[1])
            flow_bwd = resize_flow(flow_bwd, img_train.shape[0], img_train.shape[1])

            mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)
            if cnt <= 20 and check_flow(flow_bwd):
                small_cnt = small_cnt + 1
            cnt = cnt + 1
            if small_cnt >= args.little_movement_threshold:
                print("too little movements, drop")     # early check, prevent long time processing
                shutil.rmtree(output_path)
                shutil.rmtree(output_img_path)
                return 
            
            # Save flow
            np.savez(
                os.path.join(output_path, "%05d_fwd.npz" % i),
                flow=flow_fwd,
                mask=mask_fwd,
            )
            np.savez(
                os.path.join(output_path, "%05d_bwd.npz" % (i + 1)),
                flow=flow_bwd,
                mask=mask_bwd,
            )
            #! removed saving flow images
            # # Save flow_img
            # Image.fromarray(flow_viz.flow_to_image(flow_fwd)).save(
            #     os.path.join(output_img_path, "%05d_fwd.png" % i)
            # )
            # Image.fromarray(flow_viz.flow_to_image(flow_bwd)).save(
            #     os.path.join(output_img_path, "%05d_bwd.png" % (i + 1))
            # )

            # Image.fromarray(mask_fwd).save(
            #     os.path.join(output_img_path, "%05d_fwd_mask.png" % i)
            # )
            # Image.fromarray(mask_bwd).save(
            #     os.path.join(output_img_path, "%05d_bwd_mask.png" % (i + 1))
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="examples_vid/step1+2_outputs", help="Intermediate dataset output path")
    parser.add_argument("--model", help="restore RAFT checkpoint",default="weights/raft-things.pth")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--txt", type=str, default="examples_txt/flow_list.txt",help="txt file to save the flow list")
    parser.add_argument("--nomove_threshold", type=float, default=8, help="number of frames with potentially no movement, smaller means more strict")
    parser.add_argument("--little_movement_threshold", type=float, default=20, help="number of frames with potentially little movement, smaller means more strict")
    args = parser.parse_args()
    done = []
    nomove=0

    flow_done_file = args.txt

    lockpath = flow_done_file + ".lock"
    lock = FileLock(lockpath)
    if os.path.exists(flow_done_file):
        with lock:
            with open (flow_done_file, 'r') as f:
                for line in f:
                    done.append(line.strip().split(' >')[0])
            f.close()
    for dataset in microsoft_sorted(os.listdir(args.dataset_path)):
        input_path = os.path.join(args.dataset_path, dataset, "images")
        output_path = os.path.join(args.dataset_path, dataset, "flow")
        output_img_path = os.path.join(args.dataset_path, dataset, "flow_png")
        raw_files = sorted(glob.glob(input_path + '/*.jpg'), key=extract_number)
        nomove = camera_movement(raw_files[:20])
        #TODO:  problem with camera movement
        if nomove >= args.nomove_threshold:
            # with open("dynamic_outputs/dynamic_{}.txt".format(name), 'a') as f:
            print("camera maybe not moving.")
            with lock:
                with open(flow_done_file, "a") as f:
                    f.write(dataset+" >>>nomove\n")
                f.close()
            continue
        create_dir(output_path)
        create_dir(output_img_path)
        run(args, input_path, output_path, output_img_path)
        with lock:
            with open(flow_done_file, "a") as f:
                f.write(dataset+" >>>done\n")
                f.close()
    print(f"flow done. please see {flow_done_file}")
