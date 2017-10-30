from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name='tensorrec',
  packages=['tensorrec'],
  version='0.1',
  description='A TensorFlow recommendation algorithm and framework in Python.',
  long_description=long_description,
  author='James Kirk',
  author_email='james.f.kirk@gmail.com',
  url='https://github.com/jfkirk/tensorrec',
  keywords=['machine-learning', 'tensorflow', 'recommendation-system', 'python'],
  classifiers=[],
)
