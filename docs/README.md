Documentation
=============

## Setup

Documentation can be setup by executing:

```bash
$ cd docs
$ bash setup_docs.sh
```

Note: the `'Submodules'` header in `source/tensorrec.rst` was manually removed.

## Updating

Documentation can be updated as follows:

```bash
$ cd docs
$ make html
$ cd build/html
$ git checkout gh-pages
$ git add .
$ git commit -m "Update Documentation"
$ git push origin gh-pages
```
