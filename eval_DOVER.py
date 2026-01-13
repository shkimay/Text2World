import torch
import argparse
import os, re
import pickle as pkl
import pathlib
import numpy as np
import yaml
from tqdm import tqdm
from dover.models import DOVER
import pandas as pd
from torchvision import transforms as T
import subprocess
import shutil
import sys

filepath = pathlib.Path(r"C:\Users\SH\output\output")
output_dir = pathlib.Path(r"C:\Users\SH\output\dover_result")
dover_yaml = pathlib.Path(r"C:\Users\SH\PycharmProjects\PSEL\DOVER\dover.yml")
pattern = "*-*.mp4"
video_paths = sorted(filepath.glob(pattern))

def cal_dover(video: pathlib.Path,
              out_dir: pathlib.Path,
              dover_yaml: pathlib.Path | None,) -> pathlib.Path:

   eval_py = pathlib.Path(r"C:\Users\SH\PycharmProjects\PSEL\DOVER\evaluate_a_set_of_videos.py")
   output_dir.mkdir(parents=True, exist_ok=True)
   out_csv = output_dir / "eval_dover_result4.csv"

   if not eval_py.is_file():
       raise FileNotFoundError("evaluate_a_set_of_videos.py를 찾을 수 없습니다.")

   cmd = [sys.executable, str(eval_py), "-in", str(video), "-out", str(out_csv)]
   if dover_yaml:
       cmd += ["-o", str(dover_yaml)]

   print("[DOVER]", " ".join(cmd))
   env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
   res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(eval_py.parent), env=env, check=True)

   if res.returncode != 0:
       print("[DOVER-ERR]\n", res.stdout, "\n", res.stderr, sep="")
       raise RuntimeError(f"DOVER 평가 실패")
   else:
       print("[DOVER-OK]", out_csv)
   return out_csv

def main():

    dover_csv = cal_dover(filepath, output_dir, dover_yaml)

if __name__ == "__main__":
    main()