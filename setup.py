# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os.path
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(name='kauri',
        version='1.0.0',
        description="Algebraic manipulation of non-planar rooted trees in Python",
        packages=setuptools.find_packages(),
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Win32 (MS Windows)",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        url="https://github.com/daniil-shmelev/kauri",
        author="Daniil Shmelev",
        author_email="daniil.shmelev23@imperial.ac.uk",
        install_requires=['matplotlib', 'plotly', 'numpy', 'scipy', 'sympy', 'tqdm']
      )