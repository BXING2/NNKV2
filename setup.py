from setuptools import setup, find_packages

setup(
name="nnk",
version="2.0.0",
description="python package for vacancy diffusion simulation",
#packages=find_packages(),
packages=["nnk"],
classifiers=[
"Programming Language :: Python :: 3",
"License :: OSI Approved :: MIT License",
"Operating System :: OS Independent",
],
python_requires=">=3.9",
)
