from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

setup(
  name='tensorrec',
  packages=['tensorrec'],
  version='0.25.9',
  description='A TensorFlow recommendation algorithm and framework in Python.',
  author='James Kirk',
  author_email='james.f.kirk@gmail.com',
  url='https://github.com/jfkirk/tensorrec',
  keywords=['machine-learning', 'tensorflow', 'recommendation-system', 'python', 'recommender-system'],
  classifiers=[],
  install_requires=[
      "numpy==1.14.1",
      "scipy==0.19.1",
      "six==1.11.0",
      "tensorflow==1.7.0",
  ],
)
