#!/usr/bin/env python

from distutils.core import setup

setup(name='comet',
      version='1.0',
      description='packaged version of comet-commonsense library',
      author='Antoine Bosselut',
      url='https://github.com/kearnsw/comet-commonsense',
      packages=['comet',
                'comet.data',
                'comet.evaluate',
                'comet.interactive',
                'comet.models',
                'comet.train',
                'comet.utils'],
     )
