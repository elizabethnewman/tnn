from setuptools import setup, find_packages

setup(
    name='tnn',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/tnn',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='Matrix-Mimetic Tensor Neural Networks',
    install_requires=['torch==2.1.0', 'numpy==1.24.2', 'scipy==1.10.1', 'torchvision', 'matplotlib']
)
