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
    scale = self.scale

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = self.model(img_lq)
        
        output = output[..., :h_old * scale, :w_old * scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    
    return self.postprocess(output)

from torch.nn import functional as F
import math

class SwinIRUpscaler(Upscaler):
  def __init__(self, model_path, model, param_key, scale, window_size, tile=False, tile_size=256, tile_pad=32):
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key] if param_key in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(td)
    self.model = model
    self.window_size = window_size
    self.scale = scale
    self.tile = tile
    self.tile_size = tile_size
    self.tile_pad = tile_pad

  def tile_process(self, img):
      """It will first crop input images to tiles, and then process each tile.
      Finally, all the processed tiles are merged into one images.
      Modified from: https://github.com/ata4/esrgan-launcher
      """
      batch, channel, height, width = img.shape
      output_height = height * self.scale
      output_width = width * self.scale
      output_shape = (batch, channel, output_height, output_width)

      # start with black image
      output = img.new_zeros(output_shape)
      tiles_x = math.ceil(width / self.tile_size)
      tiles_y = math.ceil(height / self.tile_size)

      # loop over all tiles
      for y in range(tiles_y):
          for x in range(tiles_x):
              # extract tile from input image
              ofs_x = x * self.tile_size
              ofs_y = y * self.tile_size
              # input tile area on total image
              input_start_x = ofs_x
              input_end_x = min(ofs_x + self.tile_size, width)
              input_start_y = ofs_y
              input_end_y = min(ofs_y + self.tile_size, height)

              # input tile area on total image with padding
              input_start_x_pad = max(input_start_x - self.tile_pad, 0)
              input_end_x_pad = min(input_end_x + self.tile_pad, width)
              input_start_y_pad = max(input_start_y - self.tile_pad, 0)
              input_end_y_pad = min(input_end_y + self.tile_pad, height)

              # input tile dimensions
              input_tile_width = input_end_x - input_start_x
              input_tile_height = input_end_y - input_start_y
              tile_idx = y * tiles_x + x + 1
              input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

              # upscale tile
              output_tile = self.model(input_tile)

              # output tile area on total image
              output_start_x = input_start_x * self.scale
              output_end_x = input_end_x * self.scale
              output_start_y = input_start_y * self.scale
              output_end_y = input_end_y * self.scale

              # output tile area without padding
              output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
              output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
              output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
              output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

              # put tile into output image
              output[:, :, output_start_y:output_end_y,
                          output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                      output_start_x_tile:output_end_x_tile]
              
      return output

  def do_upscale(self, img, scale):
    img = self.preprocess(img)

    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(td)
    window_size = self.window_size
    scale = self.scale

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        if self.tile:
          output = self.tile_process(img)
        else:
          output = self.model(img)
        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return self.postprocess(output)
