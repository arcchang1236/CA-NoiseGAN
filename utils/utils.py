import cv2
import torch
import numpy as np

def read_metadata(metadata):
  meta = metadata['metadata'][0, 0]
  cam = get_cam(meta)
  bayer_pattern = get_bayer_pattern(meta)
  # We found that the correct Bayer pattern is GBRG in S6
  if cam == 'S6':
    bayer_pattern = [1, 2, 0, 1]
  bayer_2by2 = (np.asarray(bayer_pattern) + 1).reshape((2, 2)).tolist()
  wb = get_wb(meta)
  cst1, cst2 = get_csts(meta) # use cst2 for rendering
  iso = get_iso(meta)
  
  return meta, bayer_2by2, wb, cst2, iso, cam

def get_iso(metadata):
  try:
    iso = metadata['ISOSpeedRatings'][0][0]
  except:
    try:
      iso = metadata['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
    except:
      raise Exception('ISO not found.')
  return iso


def get_cam(metadata):
  model = metadata['Make'][0]
  cam_dict = {'Apple': 'IP', 'Google': 'GP', 'samsung': 'S6', 'motorola': 'N6', 'LGE': 'G4'}
  return cam_dict[model]


def get_bayer_pattern(metadata):
  bayer_id = 33422
  bayer_tag_idx = 1
  try:
    unknown_tags = metadata['UnknownTags']
    if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
      bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
    else:
      raise Exception
  except:
    try:
      unknown_tags = metadata['SubIFDs'][0, 0]['UnknownTags'][0, 0]
      if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
        bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
      else:
        raise Exception
    except:
      try:
        unknown_tags = metadata['SubIFDs'][0, 1]['UnknownTags']
        if unknown_tags[1]['ID'][0][0][0] == bayer_id:
          bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
        else:
          raise Exception
      except:
        print('Bayer pattern not found. Assuming RGGB.')
        bayer_pattern = [1, 2, 2, 3]
  return bayer_pattern


def get_wb(metadata):
  return metadata['AsShotNeutral']


def get_csts(metadata):
  return metadata['ColorMatrix1'].reshape((3, 3)), metadata['ColorMatrix2'].reshape((3, 3))


def toTensor(patch, cam):
  # Convert Bayer into BGGR
  if cam == 'IP': # RGGB
    patch = np.rot90(patch, 2)
  elif cam == 'S6': # GBRG
    patch = np.flip(patch, axis=1)
  else: # GP, N6, G4: BGGR
    patch = patch
  # Space to depth
  patch = space_to_depth(np.expand_dims(patch, axis=-1))
  # Add the batch size channel
  patch = np.expand_dims(patch, 0)
  # To tensor
  tensor = torch.from_numpy(patch.transpose((0, 3, 1, 2))).float()
  
  return tensor

    
def toPatch(tensor, cam):
  # To numpy
  patch = tensor.cpu().detach().numpy()
  # Depth to space and squeeze the batch size channel
  patch = np.squeeze(depth_to_space(np.transpose(patch[0],(1,2,0))), axis=2)
  # Conver back to original Bayer
  if cam == 'IP': # BGGR
    patch = np.rot90(patch, 2)
  elif cam == 'S6': # GBRG
    patch = np.flip(patch, axis=1)
  else: # GP, N6, G4: BGGR
    patch = patch
  
  return patch

def toTensor_nf(patch, cam):
  # Convert Bayer into RGGB
  if cam == 'IP': # RGGB
    patch = patch
  elif cam == 'S6': # GBRG
    patch = np.rot90(patch, 3)
  else: # GP, N6, G4: BGGR
    patch = np.rot90(patch, 2)
  # Space to depth
  patch = space_to_depth(np.expand_dims(patch, axis=-1))
  # Add the batch size channel
  patch = np.expand_dims(patch, 0)
  
  return patch

def toPatch_nf(patch, cam):
  # Depth to space and squeeze the batch size channel
  patch = np.squeeze(depth_to_space(patch[0]), axis=2)
  # Conver back to original Bayer
  if cam == 'IP': # RGGB
    patch = patch
  elif cam == 'S6': # GBRG
    patch = np.rot90(patch, 1)
  else: # GP, N6, G4: BGGR
    patch = np.rot90(patch, 2)
  
  return patch

def space_to_depth(x, block_size=2):
  x = np.asarray(x)
  height, width, depth = x.shape
  reduced_height = height // block_size
  reduced_width = width // block_size
  y = x.reshape(reduced_height, block_size, reduced_width, block_size, depth)
  z = np.swapaxes(y, 1, 2).reshape(reduced_height, reduced_width, -1)
  return z

def depth_to_space(x, block_size=2):
  x = np.asarray(x)
  height, width, _ = x.shape
  increased_height = height * block_size
  increased_width = width * block_size
  y = x.reshape(height, width, block_size, block_size, -1)
  z = np.swapaxes(y, 1, 2).reshape(increased_height, increased_width, -1)
  return z


def process_sidd_image(image, bayer_pattern, wb, cst, *, save_file_rgb=None):
  """Simple processing pipeline"""
  image = flip_bayer(image, bayer_pattern)
  image = stack_rggb_channels(image)
  rgb2xyz = np.array(
    [
      [0.4124564, 0.3575761, 0.1804375],
      [0.2126729, 0.7151522, 0.0721750],
      [0.0193339, 0.1191920, 0.9503041],
    ]
  )
  rgb2cam = np.matmul(cst, rgb2xyz)
  cam2rgb = np.linalg.inv(rgb2cam)
  cam2rgb = cam2rgb / np.sum(cam2rgb, axis=-1, keepdims=True)
  image_srgb = process(image, 1 / wb[0][0], 1 / wb[0][1], 1 / wb[0][2], cam2rgb)
  image_srgb = image_srgb * 255.0
  image_srgb = image_srgb.astype(np.uint8)
  image_srgb = swap_channels(image_srgb)

  if save_file_rgb:
    # Save
    cv2.imwrite(save_file_rgb, image_srgb)

  return image_srgb

def flip_bayer(image, bayer_pattern):
  if (bayer_pattern == [[1, 2], [2, 3]]):
    pass
  elif (bayer_pattern == [[2, 1], [3, 2]]):
    image = np.fliplr(image)
  elif (bayer_pattern == [[2, 3], [1, 2]]):
    image = np.flipud(image)
  elif (bayer_pattern == [[3, 2], [2, 1]]):
    image = np.fliplr(image)
    image = np.flipud(image)
  else:
    import pdb
    pdb.set_trace()
    print('Unknown Bayer pattern.')
  return image

def stack_rggb_channels(raw_image):
  """Stack the four RGGB channels of a Bayer raw image along a third dimension"""
  height, width = raw_image.shape
  channels = []
  for yy in range(2):
    for xx in range(2):
      raw_image_c = raw_image[yy:height:2, xx:width:2].copy()
      channels.append(raw_image_c)
  channels = np.stack(channels, axis=-1)
  return channels

def swap_channels(image):
  """Swap the order of channels: RGB --> BGR"""
  h, w, c = image.shape
  image1 = np.zeros(image.shape)
  for i in range(c):
    image1[:, :, i] = image[:, :, c - i - 1]
  return image1

def RGGB2Bayer(im):
  # convert RGGB stacked image to one channel Bayer
  bayer = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
  bayer[0::2, 0::2] = im[:, :, 0]
  bayer[0::2, 1::2] = im[:, :, 1]
  bayer[1::2, 0::2] = im[:, :, 2]
  bayer[1::2, 1::2] = im[:, :, 3]
  return bayer

def demosaic_CV2(rggb_channels_stack):
  # using opencv demosaic
  bayer = RGGB2Bayer(rggb_channels_stack)
  dem = cv2.cvtColor(np.clip(bayer * 16383, 0, 16383).astype(dtype=np.uint16), cv2.COLOR_BayerBG2RGB_EA)
  dem = dem.astype(dtype=np.float32) / 16383
  return dem

def apply_gains(bayer_image, red_gains, green_gains, blue_gains):
  gains = np.stack([red_gains, green_gains, green_gains, blue_gains], axis=-1)
  gains = gains[np.newaxis, np.newaxis, :]
  return bayer_image * gains

def demosaic_simple(rggb_channels_stack):
  channels_rgb = rggb_channels_stack[:, :, :3]
  channels_rgb[:, :, 0] = channels_rgb[:, :, 0]
  channels_rgb[:, :, 1] = np.mean(rggb_channels_stack[:, :, 1:3], axis=2)
  channels_rgb[:, :, 2] = rggb_channels_stack[:, :, 3]
  return channels_rgb

def apply_ccm(image, ccm):
  images = image[:, :, np.newaxis, :]
  ccms = ccm[np.newaxis, np.newaxis, :, :]
  return np.sum(images * ccms, axis=-1)

def gamma_compression(images, gamma=2.2):
  return np.maximum(images, 1e-8) ** (1.0 / gamma)

def process(bayer_images, red_gains, green_gains, blue_gains, cam2rgbs):
  bayer_images = apply_gains(bayer_images, red_gains, green_gains, blue_gains)
  bayer_images = np.clip(bayer_images, 0.0, 1.0)
  images = demosaic_CV2(bayer_images)
  images = apply_ccm(images, cam2rgbs)
  images = np.clip(images, 0.0, 1.0)
  images = gamma_compression(images)
  return images


def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
  data_range = right_edge - left_edge
  bin_width = data_range / n_bins
  if bin_edges is None:
    bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
  bin_centers = bin_edges[:-1] + (bin_width / 2.0)
  n = np.prod(data.shape)
  hist, _ = np.histogram(data, bin_edges)
  return hist / n, bin_centers

def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
  """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
  bw = 0.2 / 64
  bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
  p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
  q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
  idx = (p > 0) & (q > 0)
  p = p[idx]
  q = q[idx]
  logp = np.log(p)
  logq = np.log(q)
  kl_fwd = np.sum(p * (logp - logq))
  kl_inv = np.sum(q * (logq - logp))
  kl_sym = (kl_fwd + kl_inv) / 2.0
  return kl_fwd #, kl_inv, kl_sym