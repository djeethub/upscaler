#import ESRGAN.RRDBNet_arch as arch
from swin2sr.models.network_swin2sr import Swin2SR as Swin2SR_net

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import os
import cv2
import numpy as np
from PIL import Image
import torch

td = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Upscaler:
  def do_upscale(self, img, scale):
    return img

  def preprocess(self, img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

  def postprocess(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img
    
  def upscale(self, img, scale):
    dest_w = int(img.width * scale)
    dest_h = int(img.height * scale)

    img = self.do_upscale(img, scale)

    if img.width != dest_w or img.height != dest_h:
      img = img.resize((int(dest_w), int(dest_h)), resample=Image.LANCZOS)

    return img

class RealEsrganUpscaler(Upscaler):
  upsampler = None
  
  def __init__(self, scale, model_path, model, tile=0, tile_pad=10, pre_pad=10):
    self.upsampler = RealESRGANer(
          scale=scale,
          model_path=model_path,
          model=model,
          tile=tile,
          tile_pad=tile_pad,
          pre_pad=pre_pad,
          half=True if torch.cuda.is_available() else False,
          gpu_id=None)
    
  def do_upscale(self, img, scale):
    img = self.preprocess(img)
    img, _ = self.upsampler.enhance(img, outscale=scale)
    return self.postprocess(img)

class EsrganUpscaler(Upscaler):
  model = None

  def __init__(self, model_path, model):
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(td)
    self.model = model

  def do_upscale(self, img, scale):
    img = self.preprocess(img)

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(td)

    with torch.no_grad():
      output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output = output.astype(np.uint8)

    return self.postprocess(output)

class Swin2SrUpscaler(Upscaler):
  model = None
  scale = 0

  def __init__(self, model_path, model, param_key, scale):
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key] if param_key in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(td)
    self.model = model
    self.scale = scale

  def do_upscale(self, img, scale):
    img = self.preprocess(img)

    img_lq = img.astype(np.float32) / 255.
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(td)  # CHW-RGB to NCHW-RGB
    window_size = self.model.window_size

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = self.model(img_lq)
        
        output = output[..., :h_old * self.scale, :w_old * self.scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    
    return self.postprocess(output)

def get_upscaler(name: str) -> Upscaler:
  file_url = None
  if name == "RRDB_ESRGAN_x4":
    model = RRDBNet(3, 3, 64, 23, gc=32)
    file_url = ['https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth']
  if name == "4x-UltraSharp":
    model = RRDBNet(3, 3, 64, 23, gc=32)
    file_url = ['https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth']
  if name == "RealESRGAN_x4plus":
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
  if name == "RealESRGAN_x4plus_anime_6B":
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
  if name == "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR":
    netscale = 4
    model = Swin2SR_net(upscale=netscale, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                          mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    param_key = 'params_ema'
    file_url = ['https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth']
  if name == "Swin2SR_ClassicalSR_X2_64":
    netscale = 2
    model = Swin2SR_net(upscale=netscale, in_chans=3, img_size=64, window_size=8,
                      img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                      mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key = 'params'
    file_url = ['https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth']

  if file_url is not None:
    out_path = "/content/upscaler"
    model_path = os.path.join(out_path, name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=out_path, progress=True, file_name=None)

  upscaler = None
  if "RealESRGAN" in name:
    upscaler = RealEsrganUpscaler(netscale, model_path, model)
  elif "Swin2SR" in name:
    upscaler = Swin2SrUpscaler(model_path, model, param_key, netscale)
  else:
    upscaler = EsrganUpscaler(model_path, model)

  upscaler.name = name
  return upscaler

def get_upscaler_names():
  return ["RRDB_ESRGAN_x4",
          "4x-UltraSharp",
          "RealESRGAN_x4plus",
          "RealESRGAN_x4plus_anime_6B",
          "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR",
          "Swin2SR_ClassicalSR_X2_64"]
