import cv2, torch
import numpy as np
from PIL import Image

from realesrgan import RealESRGANer

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

from torch.nn import functional as F

class SwinIRUpscaler(Upscaler):
  def __init__(self, model_path, model, param_key):
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key] if param_key in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(td)
    self.model = model

  def do_upscale(self, img, scale):
    img = self.preprocess(img)

    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(td)
    window_size = self.model.window_size
    scale = self.model.upscale

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % self.window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % self.window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        output = self.model(img)
        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return self.postprocess(output)
