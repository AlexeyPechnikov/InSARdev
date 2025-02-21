#!/usr/bin/env python
# ----------------------------------------------------------------------------
# InSAR.dev Professional
#
# This file is part of the InSAR.dev Professional project: https://github.com/mobigroup/InSARdev-pro
#
# Copyright (c) 2025, Alexey Pechnikov
#
# This source code is licensed under a Proprietary License.
# For license terms and conditions see the LICENSE.md file provided with the source code.
# ----------------------------------------------------------------------------

from setuptools import setup
import urllib.request

def get_version():
    with open("insardev/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split('=')[1]
                version = version.replace("'", "").replace('"', "").strip()
                return version

# read the contents of local README file
#from pathlib import Path
#this_directory = Path(__file__).parent
#long_description = (this_directory / "README.md").read_text()

upstream_url = 'https://raw.githubusercontent.com/AlexeyPechnikov/pygmtsar/pygmtsar2/README.md'
response = urllib.request.urlopen(upstream_url)
long_description = response.read().decode('utf-8')

setup(
    name='insardev',
    version=get_version(),
    description='InSAR.dev (Python InSAR): Satellite Interferometry Framework Pro',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlexeyPechnikov/pygmtsar',
    author='Alexey Pechnikov',
    author_email='alexey@pechnikov.dev',
    license='Proprietary',
    packages=['insardev'],
    include_package_data=True,
    install_requires=['insardev_toolkit',
                      'xarray',
                      'numpy',
                      'numba',
                      'pandas',
                      'geopandas',
                      'distributed',
                      'dask[complete]',
                      'scipy',
                      'xgboost',
                      'cffi',
                      'scikit-learn',
                      'statsmodels>=0.14.0',
                      'matplotlib',
                      'adjustText',
                      'seaborn'
                      ],
#    extras_require={
#                      'vtk_support': ['vtk', 'panel']
#    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
    python_requires='>=3.10',
    keywords='satellite interferometry, InSAR, remote sensing, geospatial analysis, Sentinel-1, SBAS, PSI'
)
