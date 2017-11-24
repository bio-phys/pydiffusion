pydiffusion
-----------

This module is supposed to be helpful for people who are doing Molecular
Dynamics analysis. It contains a variety of small functions and wrappers around
other scientific python libraries that are useful for MD analysis. It is by no
means complete and usually implements a simple variant of algorithms. For
example the markov and transition module are thought to be easy to use functions
one can use to check if it is worth to keep looking into this type of analyzis
for a comprehensive Markov-State-Model analyzis it is better to use one of the
specialized packages like MSMBuilder.


INSTALL
-------

```
python setpy.py install
```

If you want to install the library local for your user then append the `--user`
flag. This is recommended.


DEVELOPMENT
-----------

To install the library in development mode use

```
python setup.py develop --user
```

This will create a python-package-symlink to this folder and every change you
make is directly applied to your installed package.

## Testing

In the rot folder run `python -m pytest hummer`

## Virtual Environments

If you want to seperate your experimental changes to this library from the
working installation you normally use setup a virtual environment.

```
virtualenv venv
source venv/bin/activate
python setup.py develop
```

You can deactivate the virtual environment with `deactivate`. The `setup.py`
command has only needs to be executed the first time.
