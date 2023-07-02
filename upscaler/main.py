import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.swinir_arch import SwinIR
from .swin2sr.network_swin2sr import Swin2SR as Swin2SR_net
from .esrgan.RRDBNet_arch import RRDBNet as UltraSharp_net
from .hat.hat_arch import HAT

from .upscaler import Upscaler, RealEsrganUpscaler, Swin2SrUpscaler, EsrganUpscaler, SwinIRUpscaler

def get_upscaler(name: str) -> Upscaler:
  file_url = None
  if name == "RRDB_ESRGAN_x4":
    model = RRDBNet(3, 3, 64, 23, gc=32)
    file_url = ['https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth']
  elif name == "4x-UltraSharp":
    model = UltraSharp_net(3, 3, 64, 23, gc=32)
    file_url = ['https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth']
  elif name == "RealESRGAN_x4plus":
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
  elif name == "RealESRGAN_x4plus_anime_6B":
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
  elif name == "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR":
    netscale = 4
    model = Swin2SR_net(upscale=netscale, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                          mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    param_key = 'params_ema'
    file_url = ['https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth']
  elif name == "Swin2SR_ClassicalSR_X2_64":
    netscale = 2
    model = Swin2SR_net(upscale=netscale, in_chans=3, img_size=64, window_size=8,
                      img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                      mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key = 'params'
    file_url = ['https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth']
  elif name == "Swin2SR_ClassicalSR_X4_64":
    netscale = 4
    model = Swin2SR_net(upscale=netscale, in_chans=3, img_size=64, window_size=8,
                      img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                      mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key = 'params'
    file_url = ['https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth']
  elif name == "SwinIR-L_x4_GAN":
    netscale = 4
    window_size = 8
    model = SwinIR(upscale=netscale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key = 'params_ema'
    file_url = ['https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth']
  elif name == "Real_HAT_GAN_SRx4":
    netscale = 4
    window_size = 16
    model = HAT(upscale=netscale, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
                squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
                depths=[6, 6, 6, 6, 6, 6], embed_dim=180,num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key = 'params_ema'
    file_url = ['https://huggingface.co/bullhug/test1/resolve/main/Real_HAT_GAN_SRx4.pth']

  if file_url is not None:
    out_dir = "./upscaler"
    model_path = os.path.join(out_dir, os.path.basename(file_url[0]))
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(url=url, model_dir=out_dir, progress=True, file_name=None)

  upscaler = None
  if "RealESRGAN" in name:
    upscaler = RealEsrganUpscaler(netscale, model_path, model)
  elif "Swin2SR" in name:
    upscaler = Swin2SrUpscaler(model_path, model, param_key, netscale)
  elif "SwinIR" in name:
    upscaler = SwinIRUpscaler(model_path, model, param_key, netscale, window_size)
  elif "HAT" in name:
    upscaler = SwinIRUpscaler(model_path, model, param_key, netscale, window_size, True, 256, window_size * 2)
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
          "Swin2SR_ClassicalSR_X2_64",
          "Swin2SR_ClassicalSR_X4_64",
          "SwinIR-L_x4_GAN",
          "Real_HAT_GAN_SRx4"]
