import glob
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import torchvision
import torch
import cv2
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

from einops import rearrange

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def run_maskrcnn(model, img_path):
    threshold = 0.5
    cnt = 0
    length = len(img_path)
    sample = [0, 1, 2, 3, 4, length//8, length//4, length//4 + 1, length//2, length//2 +1, length//2 + 2, length//4*3, length-1]
    for index in sample:
        img = img_path[index]
        o_image = Image.open(img).convert("RGB")
        width, height = o_image.size
        if width > height:
            intHeight = 576
            intWidth = 1024
        else:
            intHeight = 1024
            intWidth = 576

        image = o_image.resize((intWidth, intHeight), Image.LANCZOS)

        image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

        objPredictions = model([image_tensor])[0]

        for intMask in range(objPredictions["masks"].size(0)):
            if objPredictions["scores"][intMask].item() > threshold:
                # person, vehicle, accessory, animal, sports
                if objPredictions["labels"][intMask].item() == 1:  # person
                    cnt += 1
    return cnt

def white_count(img):
    pixel_count = img.size
    white_pixel_count = np.count_nonzero(img == 255)
    proportion = white_pixel_count / pixel_count
    return proportion

def central_count(img):

    image_width, image_height = img.shape[1], img.shape[0]

    central_width_start = int(image_width * 0.5)
    central_width_end = int(image_width * 0.75)
    central_height_start = int(image_height * 0.5)
    central_height_end = int(image_height * 0.75)

    central_region = img[central_height_start:central_height_end, central_width_start:central_width_end]

    white_pixel_count_central = np.count_nonzero(central_region == 255)

    proportion_central = white_pixel_count_central / central_region.size
    return proportion_central



def check_flow(folder_path):
    flow = sorted(glob.glob(os.path.join(folder_path, 'flow') + '/*bwd.npz'))
    count = 0
    for f in flow[:20]:
        npz = np.load(f)
        flow = npz['flow']
        move = abs(flow)[...,0]+ abs(flow)[...,1]
        if move.mean() <= 0.55:
            count += 1
    return count

def cal_oclidean_distance(coord1, coord2):  # N,2
    x2s,y2s = coord2[:,0], coord2[:,1]
    x1s,y1s = coord1[:,0], coord1[:,1]
    return torch.sqrt((x2s - x1s)**2 + (y2s - y1s)**2)
    

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def circle_from_three_points(p1, p2, p3):
    A = 2 * (p2[0] - p1[0])
    B = 2 * (p2[1] - p1[1])
    C = 2 * (p3[0] - p1[0])
    D = 2 * (p3[1] - p1[1])
    E = p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2
    F = p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2
    G = A * D - B * C

    if G == 0:
        return None

    center_x = (D * E - B * F) / G
    center_y = (A * F - C * E) / G
    center = np.array([center_x, center_y])
    radius = dist(center, p1)
    return center, radius

def distances_to_center(points, center):
    return np.linalg.norm(points - center, axis=1)

def ransac_circle(points, iterations=1000, min_inlier_ratio=0.5):
    best_inliers = []
    best_circle = None
    n_points = points.shape[0]
    min_inliers = int(n_points * min_inlier_ratio)
    mini_radius = 1e9
    
    for _ in range(iterations):
        try:
            sample_points = points[np.random.choice(points.shape[0], 3, replace=False)]
        except ValueError:
            continue
        circle = circle_from_three_points(sample_points[0], sample_points[1], sample_points[2])
        
        if circle is None:
            continue
        
        center, radius = circle
        distances = distances_to_center(points, center)
        inliers = points[distances <= radius]
        
        if len(inliers) >= min_inliers and mini_radius > radius:
            mini_radius = radius
            best_inliers = inliers
            best_circle = (center, radius)
    if best_circle:
        center, radius = best_circle
        return center, radius, best_inliers
    else:
        return None

def main_func(points):
    # breakpoint()
    result = ransac_circle(points, iterations=500, min_inlier_ratio=0.6)
    if result:
        center, radius, inliers = result
        return center, radius, inliers
    else:
        # print("No circle found with the required inlier ratio")
        return -1,-1,-1
def track_detect(img_path, extractor, cotracker):
    frames = []
    for img in img_path[::4]:
        frames.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    frames = np.array(frames)
    device = 'cuda'
    # breakpoint()
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
    feats0 = extractor.extract(rearrange(torch.tensor(np.array(frames[0])).cuda(), 'H W C -> C H W') / 255.)
    keypoints = feats0['keypoints'].int()
    zeros = torch.zeros_like(keypoints[...,0])
    keypoints = torch.cat([zeros.unsqueeze(-1), keypoints], dim=-1).float()
    pred_tracks, pred_visibility = cotracker(video, queries=keypoints, backward_tracking=True) # B T N 2,  B T N 1
    radius = []
    for i in range(pred_tracks.shape[2]):   # B T N 2
        tall = pred_tracks[0,:,i,:]
        mask = pred_visibility[0,:,i]  # i point, all time steps
        tall = tall[mask == 1]
        center, rad, inliers = main_func(tall.cpu().numpy())
        radius.append(rad)
    radius = sorted(torch.tensor(radius))
    count = 0
    for i in range(len(radius)):
        if radius[i] <= 20:
            count += 1
    try:
        mean = torch.tensor(radius).mean()
    except RuntimeError:
        mean = 0
    return count, mean



def check_dynamic(folder_path, output_txt, extractor=None,cotracker=None):


    if os.path.exists(os.path.join(folder_path,'epipolar_error_png')):  # initial version
        png_files = glob.glob(folder_path + '/epipolar_error_png/*.png')[:200]
    else:
        try:
            png_files = glob.glob(folder_path + '/dynamic_mask/*.png')[:200]
        except:
            print(f"Error with {folder_path}")
            return
    vid_name = folder_path.split('/')[-1]
    raw_files = sorted(glob.glob(folder_path + '/images/*.jpg'),key=extract_number)[:200]

    num_imgs = len(png_files)
    certain = 0     
    has_mask = 0
    # for file in png_files:
    centralsave = 0
    manysave = 0
    for id in sorted(png_files):

        many_motion = 0     
        central_motion = 0  

        img = Image.open(id)
        try:
            img = np.array(img)
        except:
            continue

        many_motion = white_count(img)
        central_motion = central_count(img)
        # print(many_motion, central_motion)
        if many_motion > 0.0001:
            has_mask += 1
        if many_motion >=0.12 and central_motion >= 0.35:
            certain += 2
        elif many_motion >=0.12 and central_motion < 0.2:
            certain += 0.8
        elif many_motion >= 0.05 and central_motion >= 0.2:  
            certain += 1.5
        elif many_motion <0.05 and central_motion >= 0.2:    
            certain += 1
        elif many_motion >= 0.05 and central_motion < 0.2:   
            certain += 0.5
        centralsave += central_motion
        manysave += many_motion

    count, mean = track_detect(raw_files, extractor, cotracker)

    if has_mask >= 0.6 * num_imgs:  
        print(vid_name+"too much, dynamic")
        # f.write(vid_name + '>>{}/{}_too much \n'.format(has_mask,num_imgs))  
        return
    if count >= 40 and mean <= 10:  # adjustable, count: higher more strict, mean: lower more strict
        print(vid_name+"too small movements")
        return
    else:
        with open(output_txt, 'a') as f:
            f.write(vid_name + ' >\n')
        print(vid_name+"static,saved")

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--mask_done_file", "-f", type=str, help="Folder path",default="examples_txt/mask_list.txt")
    args.add_argument("--dataset_path" , "-v", type=str,default="examples_vid/step1+2_outputs",help="Processed video clips, not the origin videos.")    # 
    args.add_argument("--txt", default="examples_txt/final.txt", type=str)
    args.add_argument("--gpu", type=str, default="0")
    args = args.parse_args()
    now = []
    output_txt = args.txt
    extractor = SuperPoint(max_num_keypoints=100).eval().cuda()  # load the extractor
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to('cuda')

    with open (args.mask_done_file, 'r') as f:
        for line in f:
            try:
                if line.strip().split(' >')[1].endswith('done'):
                    now.append(line.strip().split(' >')[0])
            except:
                pass
    if os.path.exists(output_txt):
        done = []
        with open(output_txt, 'r') as f:
            for line in f:
                done.append(line.strip().split(' >')[0])
        now = list(set(now) - set(done))
    for idx in tqdm(list(set(now))):
        vid_path = os.path.join(args.dataset_path, idx)
        check_dynamic(vid_path, output_txt=output_txt, extractor=extractor, cotracker=cotracker)
    print(f"filter done, please see {output_txt}")