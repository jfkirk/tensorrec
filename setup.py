from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    requirements = [i.strip() for i in f.read().splitlines()]

setup(
    name='tensorrec',
    packages=['tensorrec'],
    version='0.25.5',
    description='A TensorFlow recommendation algorithm and framework in Python.',
    author='James Kirk',
    author_email='james.f.kirk@gmail.com',
    url='https://github.com/jfkirk/tensorrec',
    install_requires=requirements,
    keywords=[
        'machine-learning', 'tensorflow',
        'recommendation-system', 'python',
        'recommender-system'
    ],
    classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7'
        'Programming Language :: Python :: 3.6'
    ],
)
