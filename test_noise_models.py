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

from models.noise_model import Generator

from utils.utils import toPatch
from utils.utils import toTensor
from utils.utils import toPatch_nf
from utils.utils import toTensor_nf
from utils.utils import read_metadata
from utils.utils import process_sidd_image
from utils.utils import cal_kld

######################### Noise Flow #########################
# If you want to get the results of Noise Flow,
# you should install tensorflow 1.13.0 and tensorflow-probability 0.5.0  
import tensorflow as tf
from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
##############################################################

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
  folder_mat, folder_png = cfg['noise_model']['sample_dir_mat'], cfg['noise_model']['sample_dir_png']
  if os.path.exists(folder_mat):
    shutil.rmtree(folder_mat)
  os.makedirs(folder_mat, exist_ok=True)
  if os.path.exists(folder_png):
    shutil.rmtree(folder_png)
  os.makedirs(folder_png, exist_ok=True)
  # Load our model
  model_ours = Generator()
  checkpoint_file = cfg['noise_model']['checkpoint_ours']
  if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model_ours.load_state_dict(checkpoint["net_G_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_kld = checkpoint["best_kld"]
    print("Ours Loaded [Epoch: %d] [KLD: %.8f]"%(start_epoch, best_kld))
    # Evaluation mode
    model_ours.eval()
  else:
    assert 0, "No checkpoint found at '%s'"%(best_ckpt_nm)
  
  ######################### Noise Flow #########################
  ## Load Noise Flow pre-trained model
  tf.reset_default_graph()
  noise_flow = NoiseFlowWrapper(cfg['noise_model']['checkpoint_dir_nf'],cfg['noise_model']['patch_size'])
  ##############################################################

  # For each test data
  for data_idx, data_file in enumerate(glob.glob(os.path.join(cfg['data_dir'],'*[!_meta].mat'))):
    # Load clean, noisy, meta data
    data = sio.loadmat(data_file)
    clean, noisy = data['clean'], data['noisy']
    metadata = sio.loadmat('%s_meta.mat' % data_file[:-4])
    meta, bayer_2by2, wb, cst2, iso, cam = read_metadata(metadata)
    beta1, beta2 =  meta['UnknownTags'][7, 0][2][0][0:2]
    print(beta1, beta2)
    
    # For each sample
    ps = cfg['noise_model']['patch_size']
    for sample_idx in range(cfg['noise_model']['sample_amount']):
      # Log
      print('%s [%04d] [ISO %04d]' %(os.path.basename(data_file), sample_idx, iso))
      # Crop patch
      x = random.randrange(0, clean.shape[0]-ps[0], 2) # 2 for Bayer pattern
      y = random.randrange(0, clean.shape[1]-ps[1], 2) # 2 for Bayer pattern
      clean_patch = clean[x:x+ps[0], y:y+ps[1]]
      noisy_patch = noisy[x:x+ps[0], y:y+ps[1]]
      # G
      g_patch = np.random.normal(0, math.sqrt(beta1), clean_patch.shape)
      # PG
      beta_y = np.sqrt(beta1 * clean_patch + beta2)
      pg_patch = beta_y * np.random.normal(0, 1, clean_patch.shape)
      # Patch to Tensor
      clean_tensor = toTensor(clean_patch, cam)
      noisy_tensor = toTensor(noisy_patch, cam)
      pg_tensor = toTensor(pg_patch, cam)
      # Ours
      ## For easy inference, we use the corresponding noisy image as input
      ## You can replace it with arbitrary noisy image from same camera to get similar performance
      ours_tensor, _, _, _ = model_ours(pg_tensor, clean_tensor, noisy_tensor, noisy_tensor, noisy_tensor)
      # Tensor to Patch
      ours_patch = toPatch(ours_tensor, cam)
      # Noisy with synthetic noise
      noisyg_patch = np.clip(clean_patch + g_patch, 0.0, 1.0)
      noisypg_patch = np.clip(clean_patch + pg_patch, 0.0, 1.0)
      noisyours_patch = np.clip(clean_patch + ours_patch, 0.0, 1.0)
      g_patch = noisyg_patch - clean_patch
      pg_patch = noisypg_patch - clean_patch
      ours_patch = noisyours_patch - clean_patch
      # Log (KL divergence)
      noise_patch = noisy_patch - clean_patch
      print("[KLD] G: %.4f | PG: %.4f | Ours: %.4f"
              %(cal_kld(noise_patch, g_patch), cal_kld(noise_patch, pg_patch), cal_kld(noise_patch, ours_patch)))
      # Save output mat
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_g.mat' % (data_idx, sample_idx)), {'data': g_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_pg.mat' % (data_idx, sample_idx)), {'data': pg_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_ours.mat' % (data_idx, sample_idx)), {'data': ours_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_clean.mat' % (data_idx, sample_idx)), {'data': clean_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_noisy.mat' % (data_idx, sample_idx)), {'data': noisy_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_noisyg.mat' % (data_idx, sample_idx)), {'data': noisyg_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_noisypg.mat' % (data_idx, sample_idx)), {'data': noisypg_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_noisyours.mat' % (data_idx, sample_idx)), {'data': noisyours_patch, 'metadata': meta})
      # Save output png
      ## In visualization of noise, we use the Matlab instead to generate the results in our paper
      g_patch_srgb = process_sidd_image(g_patch, bayer_2by2, wb, cst2)
      pg_patch_srgb = process_sidd_image(pg_patch, bayer_2by2, wb, cst2)
      ours_patch_srgb = process_sidd_image(ours_patch, bayer_2by2, wb, cst2)
      noise_patch_srgb = process_sidd_image(noise_patch, bayer_2by2, wb, cst2)
      clean_patch_srgb = process_sidd_image(clean_patch, bayer_2by2, wb, cst2)
      noisy_patch_srgb = process_sidd_image(noisy_patch, bayer_2by2, wb, cst2)
      noisyg_patch_srgb = process_sidd_image(noisyg_patch, bayer_2by2, wb, cst2)
      noisypg_patch_srgb = process_sidd_image(noisypg_patch, bayer_2by2, wb, cst2)
      noisyours_patch_srgb = process_sidd_image(noisyours_patch, bayer_2by2, wb, cst2)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_g.png' % (data_idx, sample_idx)), g_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_pg.png' % (data_idx, sample_idx)), pg_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_ours.png' % (data_idx, sample_idx)), ours_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_noise.png' % (data_idx, sample_idx)), noise_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_clean.png' % (data_idx, sample_idx)), clean_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_noisy.png' % (data_idx, sample_idx)), noisy_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_noisyg.png' % (data_idx, sample_idx)), noisyg_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_noisypg.png' % (data_idx, sample_idx)), noisypg_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_noisyours.png' % (data_idx, sample_idx)), noisyours_patch_srgb)
      
      ######################### Noise Flow #########################
      ## The Bayer of our model is BGGR, however, Noise Flow is RGGB
      clean_tensor_nf = toTensor_nf(clean_patch, cam)
      ## NF
      cam_id = {'IP': 0, 'GP': 1, 'S6': 2, 'N6': 3, 'G4': 4}
      nf_tensor = noise_flow.sample_noise_nf(clean_tensor_nf, 0.0, 0.0, iso, cam_id[cam])
      ## Tensor to Patch
      nf_patch = toPatch_nf(nf_tensor, cam)
      ## Noisy with synthetic noise
      noisynf_patch = np.clip(clean_patch + nf_patch, 0.0, 1.0)
      nf_patch = noisynf_patch - clean_patch
      ## Log (KL divergence)
      print("[KLD] NF: %.4f"%(cal_kld(noise_patch, nf_patch)))
      ## Save output mat
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_nf.mat' % (data_idx, sample_idx)), {'data': nf_patch, 'metadata': meta})
      sio.savemat(os.path.join(folder_mat, '%03d_%03d_noisynf.mat' % (data_idx, sample_idx)), {'data': noisynf_patch, 'metadata': meta})
      ## Save output png
      nf_patch_srgb = process_sidd_image(nf_patch, bayer_2by2, wb, cst2)
      noisynf_patch_srgb = process_sidd_image(noisynf_patch, bayer_2by2, wb, cst2)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_nf.png' % (data_idx, sample_idx)), nf_patch_srgb)
      cv2.imwrite(os.path.join(folder_png, '%03d_%03d_noisynf.png' % (data_idx, sample_idx)), noisynf_patch_srgb)
      ##############################################################
    
if __name__ == '__main__':
  main()

