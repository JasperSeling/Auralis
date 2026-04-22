from pathlib import Path

from setuptools import setup, find_packages
import sys
import platform
import warnings


def check_platform():
    """Warn (do not fail) when installing on non-Linux platforms.

    vLLM's GPU runtime requires Linux, but developers often install Auralis
    on Windows/macOS for editing, type-checking, and unit tests that do not
    exercise vLLM. Emitting a warning keeps ``pip install`` functional on
    those hosts while making the limitation explicit.
    """
    if sys.platform not in ('linux', 'linux2'):
        warnings.warn(
            f"Auralis' runtime dependency vLLM only supports Linux; "
            f"current platform is {platform.system()}. Installation will "
            f"proceed for local development, but inference will not work "
            f"outside a Linux host with a CUDA GPU.",
            stacklevel=2,
        )


check_platform()
setup(
    name='auralis',
    version='0.3.0',
    description='This is a faster implementation for TTS models, to be used in highly async environment',
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    url='https://github.com/astramind.ai/auralis',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={
            'console_scripts': [
                'auralis.openai=auralis.entrypoints.oai_server:main',
            ],
        },
    install_requires=[
        "aiofiles",
        "beautifulsoup4",
        "cachetools",
        "colorama",
        "cutlet",
        "EbookLib",
        "einops",
        "ffmpeg",
        "fsspec",
        "hangul_romanize",
        "huggingface_hub",
        "ipython",
        "librosa",
        "networkx",
        "num2words",
        "opencc",
        "packaging",
        "pyloudnorm",
        "pytest",
        "pypinyin",
        "safetensors>=0.4.5",
        "sounddevice",
        "soundfile",
        "spacy==3.7.5",
        "setuptools",
        "torchaudio",
        "tokenizers",
        "transformers",
        "vllm>=0.14.0",
        "nvidia-ml-py",
        "numpy>=1.26",
        "langid"

    ],
    extras_require={
        'docs': [
            'mkdocs>=1.4.0',
            'mkdocs-material>=9.0.0',
            'mkdocstrings>=0.20.0',
            'mkdocstrings-python>=1.0.0',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
