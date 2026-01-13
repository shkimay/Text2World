from lavis.models import load_model_and_preprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

filepath = Path(r"C:\Users\SH\output\output")
pattern = "*-*.mp4"
video_paths = sorted(filepath.glob(pattern))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_image_text_matching", model_type="base", is_eval=True, device=device)

def frames(video_path:Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Wrong: {video_path}")
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, n-1, 92, dtype=int)
    frames=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def vid_score(video_path, frame, caption: str, k_ratio: float = 0.2):
    image = torch.stack([vis_processors["eval"](im) for im in frame]).to(device)
    text_inputs = [txt_processors["eval"](caption)]*len(frame)

    with torch.inference_mode():
        sample = model({"image": image, "text_input": text_inputs},match_head="itm")
        prob = torch.softmax(sample, dim=1)[:,1]

    score_mean = prob.mean().item()
    score_boun = prob.topk(max(1, int(round(prob.numel() * k_ratio)))).values.mean().item()
    score_max = prob.max().item()

    print(f"{video_path.name}: ITM prob (mean of {len(frame)} frames) = {score_mean:.4f} (max={score_max:.4f})")
    return score_mean,score_boun,score_max

def pairs_(video_path):
    pairs =[]
    for v in video_path:
        t = v.with_suffix(".txt")
        cap = t.read_text(encoding="utf-8").strip()
        pairs.append((v,cap))
    return pairs
ref_pairs = pairs_(video_paths)

frames_cache = {v: frames(v) for v, _ in ref_pairs}

pos_means, pos_topk, pos_max = [], [], []
for v, cap in ref_pairs:
    m, tk, mx = vid_score(v, frames_cache[v], cap)
    pos_means.append(m); pos_topk.append(tk); pos_max.append(mx)

neg_means, neg_topk, neg_max = [], [], []
caps_shift = [c for _, c in ref_pairs[1:]] + [ref_pairs[0][1]]  # rotate
for (v, _), cap_wrong in zip(ref_pairs, caps_shift):
    m, tk, mx = vid_score(v, frames_cache[v], cap_wrong)
    neg_means.append(m); neg_topk.append(tk); neg_max.append(mx)

def summary(name, arr):
    arr = np.array(arr)
    print(f"{name}: n={arr.size} | mean={arr.mean():.4f} | median={np.median(arr):.4f} | p95={np.quantile(arr,0.95):.4f} | p99={np.quantile(arr,0.99):.4f}")

print("== Matched vs Shuffled (mean of frame probs) ==")
summary("POS(mean)", pos_means)
summary("NEG(mean)", neg_means)

print("\n== Top-20% mean ==")
summary("POS(topk)", pos_topk)
summary("NEG(topk)", neg_topk)

print("\n== Max ==")
summary("POS(max)", pos_max)
summary("NEG(max)", neg_max)

thr = float(np.quantile(neg_means, 0.95))
print(f"\n[Suggestion] Threshold@~5% FPR (mean-based): {thr:.4f}")