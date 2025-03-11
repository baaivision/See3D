import os
import imageio
from tqdm import tqdm
import cv2
import shutil
import torchvision
from PIL import Image
import argparse
try:
    import jsonlines
except:
    os.system('pip install jsonlines')
    import jsonlines
import numpy as np
from utils.flow_utils import *

def run_maskrcnn(model, img):
    threshold = 0.5
    cnt = 0
    # o_image = Image.open(img).convert("RGB")
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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', type=int,default=0)
    args.add_argument('--input_dir', type=str, default='examples', help="input video folder")
    args.add_argument('--output_dir', type=str, default='examples_vid/step1+2_outputs', help="output folder to store frames")
    args.add_argument('--txt', type=str, default="examples_txt/all_video_list.txt", help="txt file to store processed video names,use it for the following steps.")
    args.add_argument('--human_threshold', type=float, default=10, help="number of frames with human, if more than this threshold, remove the video")
    
    args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    videoname = 'alls'
    video_folder = args.input_dir
    videos_list = microsoft_sorted(os.listdir(video_folder))

    duration = 150    
    sep = 2 #  2 frame per second``
    output_path = args.output_dir
    txt_file = args.txt

    netMaskrcnn = (
        torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        .cuda()
        .eval()
        )
    done_list = []
    if os.path.exists(txt_file):
        with open (txt_file, 'r') as f:
            for line in f:
                done_list.append(line.strip())
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(txt_file.split('/')[0], exist_ok=True)
    for filename in tqdm(videos_list):

        if filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, filename)
            video_name = filename.split('/')[-1].split('.')[0]
            if filename in done_list:
                print("skipping " + video_name)
                continue
            with open (txt_file, 'a+') as f:
                f.write(filename + '\n')
                f.close()

            frame_folder = os.path.join(output_path, video_name)
            frame_folder = os.path.join(frame_folder, 'images')
            if os.path.exists(os.path.join(output_path, video_name)) and not os.path.getsize(os.path.join(output_path, video_name)):
                print("skipping " + video_name)
                continue
            os.makedirs(frame_folder, exist_ok=True)
            
            reader = imageio.get_reader(video_path, format='ffmpeg')
            

            cnt = 0
            thres = 0
            for frame_count, frame in tqdm(enumerate(reader)):
                if cnt <= 20:
                    # turn into Image
                    o_image = Image.fromarray(frame)
                    thres = thres + run_maskrcnn(netMaskrcnn, o_image)
                    if thres > args.human_threshold:
                        print("human detected, remove!")
                        shutil.rmtree(os.path.join(output_path, video_name))
                        break
                    cnt += 1
                current_time = reader.get_meta_data()['duration'] * frame_count / (reader.get_meta_data()['fps'] + 1e-9)
                if current_time >= duration: break
                if frame_count % sep == 0: 
                    
                    frame_filename = f"frame_{frame_count}.jpg"
                    frame_path = os.path.join(frame_folder, frame_filename)
                    
                    H,W = frame.shape[:2]
                    if H > 2160 or W > 2160:    # 4k
                        frame = cv2.resize(frame, (W // 8, H // 8))
                    elif H >1080 or W > 1080:   # 
                        frame = cv2.resize(frame, (W // 4, H // 4))
                    elif H > 720 or W > 720:
                        frame = cv2.resize(frame, (W // 3, H // 3))

                    imageio.imwrite(frame_path, frame)

    print(f"All videos processed done. please see {output_path}")