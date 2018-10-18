from pip.req import parse_requirements
from pip.download import PipSession
from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

install_reqs = parse_requirements(path.join(here, 'requirements.txt'), session=PipSession())
reqs = [str(ir.req) for ir in install_reqs]

setup(
  name='tensorrec',
  packages=['tensorrec'],
  version='0.25.5',
  description='A TensorFlow recommendation algorithm and framework in Python.',
  author='James Kirk',
  author_email='james.f.kirk@gmail.com',
  url='https://github.com/jfkirk/tensorrec',
  keywords=['machine-learning', 'tensorflow', 'recommendation-system', 'python', 'recommender-system'],
  classifiers=[],
  install_requires=reqs,
)
