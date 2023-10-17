import io
import os
import artcraft
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='UTF-8') as f:
    long_description = '\n' + f.read()

setup(
    name="artcraft",
    version=artcraft.__version__,
    entry_points={
        'console_scripts': [
            'artcraft=artcraft.ui:launch',
        ],
    },
    packages=find_packages(),
    license="Apache License 2.0",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords=['diffusers', 'stable diffusion'],
    author="Yajun",
    author_email="skyblueice234@gmail.com",
    description="Image generation based on diffusers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yinyajun/artcraft',
    install_requires=[
        'gradio==3.41.2',
        'huggingface_hub==0.16.4',
        'modelscope==1.8.3',
        'Pillow==10.0.0',
        'opencv-python==4.8.0.76',
        'diffusers==0.20.0',
        'accelerate==0.21.0',
        'torch==2.0.1',
        'transformers==4.31.0'
    ],
    python_requires='>=3.9'
)
