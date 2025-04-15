import os.path
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(name='kauri',
        version='0.1.0',
        description="Algebraic manipulation of planar rooted trees in Python",
        packages=['kauri'],
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
        install_requires=['matplotlib', 'numpy', 'scipy', 'sympy']
      )