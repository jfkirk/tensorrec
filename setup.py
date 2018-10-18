from setuptools import setup

from os import path

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

here = path.abspath(path.dirname(__file__))
install_reqs = parse_requirements(path.join(here, 'requirements.txt'))

setup(
  name='tensorrec',
  packages=['tensorrec'],
  version='0.25.6',
  description='A TensorFlow recommendation algorithm and framework in Python.',
  author='James Kirk',
  author_email='james.f.kirk@gmail.com',
  url='https://github.com/jfkirk/tensorrec',
  keywords=['machine-learning', 'tensorflow', 'recommendation-system', 'python', 'recommender-system'],
  classifiers=[],
  install_requires=install_reqs,
)
