import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name = "nwtools",
    version = "0.1.0",
    description = ("Analysis tools for networks"),
    license = "Apache 2.0",
    keywords = "Python",
    url = "https://github.com/dafnevk/network-tools",
    packages=find_packages(),
    install_requires=required,
    long_description=read('README.md'),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
    ],
    setup_requires=["pytest-runner", ],
    tests_require=["pytest", ],
)

