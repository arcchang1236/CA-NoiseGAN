import os
import cv2
import math
import glob
import yaml
import random
import shutil
import argparse
import numpy as np
import scipy.io as sio

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch

from models.denoiser import DnCNN

from utils.utils import toPatch
from utils.utils import toTensor
from utils.utils import read_metadata
from utils.utils import process_sidd_image

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True)
  return parser.parse_args()

def main():
  # Params
  args = parse_args()
  with open(args.config, 'rt') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
  # Create output directory
  folder_mat, folder_png = cfg['denoiser']['sample_dir_mat'], cfg['denoiser']['sample_dir_png']
  if os.path.exists(folder_mat):
    shutil.rmtree(folder_mat)
  os.makedirs(folder_mat, exist_ok=True)
  if os.path.exists(folder_png):
    shutil.rmtree(folder_png)
  os.makedirs(folder_png, exist_ok=True)
  # Create model dict
  model_dict = {}
  # For each Denoiser
  for checkpoint_idx, checkpoint_file in enumerate(glob.glob(os.path.join(cfg['denoiser']['checkpoint_dir'], '*.pth'))):
    # Load checkpoint
    if os.path.isfile(checkpoint_file):
      key = os.path.basename(checkpoint_file)[:-4]
      model_dict[key] = DnCNN()
      checkpoint = torch.load(checkpoint_file)
      model_dict[key].load_state_dict(checkpoint["state_dict"])
      # Evaluation mode
      model_dict[key].eval()
    else:
      assert 0, "No checkpoint found at '%s'"%(best_ckpt_nm)
  # For each test data
  for data_idx, data_file in enumerate(glob.glob(os.path.join(cfg['data_dir'],'*[!_meta].mat'))):
    # Load clean, noisy, meta data
    data = sio.loadmat(data_file)
    clean, noisy = data['clean'], data['noisy']
    metadata = sio.loadmat('%s_meta.mat' % data_file[:-4])
    beta1, beta2 =  metadata['metadata'][0, 0]['UnknownTags'][7, 0][2][0][0:2]
    meta, bayer_2by2, wb, cst2, iso, cam = read_metadata(metadata)
    # For each sample
    ps = cfg['denoiser']['patch_size']
    for sample_idx in range(cfg['denoiser']['sample_amount']):
      # Log
      print('%s [%04d] [ISO %04d]' %(os.path.basename(data_file), sample_idx, iso))
      # Crop patch
      x = random.randrange(0, clean.shape[0]-ps[0], 2) # 2 for Bayer pattern
      y = random.randrange(0, clean.shape[1]-ps[1], 2) # 2 for Bayer pattern
      clean_patch = clean[x:x+ps[0], y:y+ps[1]]
      noisy_patch = noisy[x:x+ps[0], y:y+ps[1]]
      # Patch to Tensor
      noisy_tensor = toTensor(noisy_patch, cam)
      # Run all detected denoisers
      for key,_ in model_dict.items():
        # Model
        output_tensor = model_dict[key](noisy_tensor)
        # Tensor to Patch
        output_patch = toPatch(output_tensor, cam)
        # PSNR and SSIM
        psnr = peak_signal_noise_ratio(clean_patch, output_patch, data_range=1)
        ssim = structural_similarity(clean_patch, output_patch)
        # Log
        print("%10s PSNR: %.4f | SSIM: %.4f"%(key, psnr, ssim))
        # Save output mat
        save_name = '%03d_%03d_%s.mat' % (data_idx, sample_idx, key)
        save_file = os.path.join(folder_mat, save_name)
        sio.savemat(save_file, {'data': output_patch, 'metadata': meta})
        # Save output png
        output_patch_srgb = process_sidd_image(output_patch, bayer_2by2, wb, cst2)
        save_name = '%03d_%03d_%s.png' % (data_idx, sample_idx, key)
        save_file = os.path.join(folder_png, save_name)
        cv2.imwrite(save_file, output_patch_srgb)
      # Save clean and noisy mat
      save_name = '%03d_%03d_clean.mat' % (data_idx, sample_idx)
      save_file = os.path.join(folder_mat, save_name)
      sio.savemat(save_file, {'data': clean_patch, 'metadata': meta})
      save_name = '%03d_%03d_noisy.mat' % (data_idx, sample_idx)
      save_file = os.path.join(folder_mat, save_name)
      sio.savemat(save_file, {'data': noisy_patch, 'metadata': meta})
      # Save clean and noisy png
      clean_patch_srgb = process_sidd_image(clean_patch, bayer_2by2, wb, cst2)
      save_name = '%03d_%03d_clean.png' % (data_idx, sample_idx)
      save_file = os.path.join(folder_png, save_name)
      cv2.imwrite(save_file, clean_patch_srgb)
      noisy_patch_srgb = process_sidd_image(noisy_patch, bayer_2by2, wb, cst2)
      save_name = '%03d_%03d_noisy.png' % (data_idx, sample_idx)
      save_file = os.path.join(folder_png, save_name)
      cv2.imwrite(save_file, noisy_patch_srgb)
    
if __name__ == '__main__':
  main()

