from setuptools import find_packages
from numpy.distutils.core import Extension, setup
import sys
import six
import os.path as osp

parseghcnrow = Extension(
    name='gwgen._parseghcnrow', sources=[
        osp.join('gwgen', 'mo_parseghcnrow.f90')])
parseeecra = Extension(
    name='gwgen._parseeecra', sources=[osp.join('gwgen', 'mo_parseeecra.f90')],
    f2py_options=['only:', 'parse_file', 'extract_data', ':'],
    extra_f90_compile_args = ["-fopenmp"], extra_link_args = ["-lgomp"])

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []


def readme():
    with open('README.rst') as f:
        return f.read()


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('gwgen')
    config.add_data_dir(osp.join('gwgen', 'src'))
    config.add_data_dir(osp.join('gwgen', 'data'))

    return config

install_requires = ['f90nml', 'psyplot', 'scipy', 'sqlalchemy', 'psycopg2',
                    'statsmodels', 'docrep']

if six.PY2:
    install_requires.append('argparse')


setup(name='gwgen',
      version='0.0.1.dev0',
      description='Python package for interactive data visualization',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: Unix',
        'Operating System :: MacOS',
      ],
      keywords='wgen weathergen ghcn eecra richardson geng',
      url='https://github.com/Chilipp/gwgen',
      author='Philipp Sommer',
      author_email='philipp.sommer@unil.ch',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*', 'examples']),
      install_requires=install_requires,
      package_data={'gwgen': [
          'gwgen/src/*',
          'gwgen/data/*',
          ]},
      include_package_data=True,
      setup_requires=pytest_runner,
      tests_require=['pytest'],
      entry_points={'console_scripts': ['gwgen=gwgen.main:main']},
      zip_safe=False,
      ext_modules=[parseghcnrow, parseeecra],
      configuration=configuration)
