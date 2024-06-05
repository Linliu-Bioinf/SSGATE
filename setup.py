#!/bin/env python3

"""
__date__ = '2024-01-24'
__author__ = 'liulin4@genomics.cn'
"""

import sys
from setuptools import setup, find_packages
from pathlib import Path

_version_ = "1.0.0"


setup(
    name = "SSGATE",
    version = _version_,
    author="liulin4",
    author_email = "liulin4@genomics.cn",
    description = "Integrative anaysis of spatial multi-omics",
    keywords=["Spatial_transcriptomics", "Stereo-CITE-seq", "bioinformatics"],
    packages = find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
         'Programming Language :: Python :: 3.9'
        'Environment :: Console'
    ],
    install_requires = [
    ]
)


