from setuptools import setup

setup(
    name='tinytorchtest',
    version='0.7',
    packages=['tinytorchtest'],
    license='GNU General Public License v3 or later (GPLv3+)',
    long_description=open('README.md').read(),
    install_requires=['torch'],
)
