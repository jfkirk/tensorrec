# TensorRec
A TensorFlow recommendation algorithm and framework in Python.

[![PyPI version](https://badge.fury.io/py/tensorrec.svg)](https://badge.fury.io/py/tensorrec) [![Build Status](https://travis-ci.org/jfkirk/tensorrec.svg?branch=master)](https://travis-ci.org/jfkirk/tensorrec) [![Gitter chat](https://badges.gitter.im/tensorrec/gitter.png)](https://gitter.im/tensorrec)

## What is TensorRec?
TensorRec is a Python recommendation system that allows you to quickly develop recommendation algorithms and customize them using TensorFlow.

TensorRec lets you to customize your recommendation system's embedding functions and loss functions while TensorRec handles the data manipulation, scoring, and ranking to generate recommendations.

A TensorRec system consumes three pieces of data: `user_features`, `item_features`, and `interactions`. It uses this data to learn to make and rank recommendations.

For an overview of TensorRec and its usage, please see the [wiki.](https://github.com/jfkirk/tensorrec/wiki)

For more information, and for an outline of this project, please read [this blog post](https://medium.com/@jameskirk1/tensorrec-a-recommendation-engine-framework-in-tensorflow-d85e4f0874e8).

## Quick Start
TensorRec can be installed via pip:
```pip install tensorrec```
