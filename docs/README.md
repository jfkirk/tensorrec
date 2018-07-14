Documentation
=============

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
