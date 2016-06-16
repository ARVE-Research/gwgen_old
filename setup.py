from setuptools import setup, find_packages
import sys

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []


def readme():
    with open('README.rst') as f:
        return f.read()


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
#      url='https://github.com/Chilipp/psyplot',
      author='Philipp Sommer',
      author_email='philipp.sommer@unil.ch',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*', 'examples']),
      install_requires=[
          'numpy',
          'PyYAML'
      ],
      package_data={'gwgen': ['gwgen/src/*.f90']},
      include_package_data=True,
      setup_requires=pytest_runner,
      tests_require=['pytest'],
      entry_points={'console_scripts': ['gwgen=gwgen.main:main']},
      zip_safe=False)
