import torch
import lpips
from IPython import embed
import os
import pandas
import glob
import cv2
from pathlib import Path
import shutil
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import re
from typing import Dict, List, Iterable, Union, Tuple
import csv
import random

Device = "cuda" if torch.cuda.is_available() else "cpu"
filepath = Path(r"C:\Users\SH\output\output")
gen_video = glob.glob(os.path.join(str(filepath), f"*-*.mp4"))
gen_image = Path(r"C:\Users\SH\output\gen_image")

def extract_frames(video_path: Union[str,Path], out_dir:Union[str,Path], num_frames = 30):
    vid_path = Path(video_path)
    out = Path(out_dir)
    out_p = out / vid_path.stem
    out_p.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"비디오를 열 수 없음:{video_path}")
        return []

    src_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not src_fps or src_fps <= num_frames:
        target_frames = list(range(src_fps))
    else:
        target_frames = random.sample(range(src_fps), num_frames)
        target_frames.sort()

    saved=[]
    frame_idx = 0
    save_idx=0
    next_save_at = 0 if num_frames <= 0 else int(round(save_idx*src_fps/num_frames))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_frames:
            out_path = out_p / f"{len(saved):05d}.png"
            if cv2.imwrite(str(out_path), frame):
                saved.append(str(out_path))
            else:
                print(f"프레임 저장 실패: {out_path}")
        frame_idx += 1
    cap.release()
    return saved

_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,)*3,(0.5,)*3)
])

def load_tensor(path: str, device=Device):
    img=Image.open(path).convert("RGB")
    t=_tf(img).unsqueeze(0).to(Device)
    return t

def lpips_video_mean(gen_videos):

    loss_fn=lpips.LPIPS(net='alex').to(Device).eval()
    device = next(loss_fn.parameters()).device

    results = {}

    CROSS_WRAP = False

    with torch.no_grad():
        for idx, vid in tqdm(enumerate(gen_videos)):
            frames = extract_frames(vid, gen_image)
            pairs = list(zip(frames[:-1], frames[1:]))
            n = len(pairs)
            total = 0

            for t1, t2 in pairs:
                s0 = load_tensor(t1, device=device)
                s1 = load_tensor(t2, device=device)
                d = loss_fn(s0, s1)
                d_val = float(d.mean().item())
                total += d_val

            dmean = total / n
            results[f"{idx+1:03d}"] = dmean

        print(f"calculated lpips: {len(results)}")
    return results

if __name__ == "__main__":
    gen_videos = sorted(gen_video)
    scores = lpips_video_mean(gen_videos)

    for k, v in scores.items():
        print(f"{k}\tLIPIPS(mean): {v:.6f}")
    with open('LPIPS_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "lpips_mean"])  # 헤더
        for k, v in scores.items():
            writer.writerow([k, v])

