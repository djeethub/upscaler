from setuptools import setup, find_packages

setup(
    name='upscaler',
    version='0.0.9',
    description='upscaler',
    url='https://github.com/djeethub/upscaler.git',
    packages=find_packages(),
    install_requires=[
      'torch', 'basicsr', 'gfpgan', 'realesrgan', 'timm'
    ],
)