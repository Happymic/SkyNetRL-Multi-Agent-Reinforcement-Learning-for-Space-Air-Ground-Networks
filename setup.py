#!/usr/bin/env python3
"""
Setup script for SkyNetRL package
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="skynetrl",
    version="2.0.0",
    description="Optimized Multi-Agent Reinforcement Learning for Space-Air-Ground Networks with Hierarchical Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Chenxu Li",
    author_email="mcl123@ic.ac.uk",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement-learning, multi-agent, attention, space-air-ground, networks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8, <4",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "gymnasium>=0.28.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "pylint>=2.15",
            "mypy>=1.0",
        ],
        "full": [
            "wandb>=0.12.0",
            "dash>=2.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "skynetrl-train=train:main",
            "skynetrl-demo=examples.quick_start_example:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/michaelli/SkyNetRL-Multi-Agent-Reinforcement-Learning-for-Space-Air-Ground-Networks/issues",
        "Source": "https://github.com/michaelli/SkyNetRL-Multi-Agent-Reinforcement-Learning-for-Space-Air-Ground-Networks",
        "Documentation": "https://github.com/michaelli/SkyNetRL-Multi-Agent-Reinforcement-Learning-for-Space-Air-Ground-Networks/blob/main/README.md",
    },
)