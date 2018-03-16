from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

setup(
  name='tensorrec',
  packages=['tensorrec'],
  version='0.22',
  description='A TensorFlow recommendation algorithm and framework in Python.',
  author='James Kirk',
  author_email='james.f.kirk@gmail.com',
  url='https://github.com/jfkirk/tensorrec',
  keywords=['machine-learning', 'tensorflow', 'recommendation-system', 'python', 'recommender-system'],
  classifiers=[],
)
