.. gwgen documentation master file, created by
   sphinx-quickstart on Mon Jul 20 18:01:33 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GWGEN: A global weather generator for daily data
================================================

Welcome! This synthesis of FORTRAN and Python is a globally applicable
weather generator inspired by the original WGEN weather generator of
[Richardson1981]_ and parameterized through a global dataset of [GHCN]_ and
[EECRA]_ data. A paper with the scientific documentation is in progress, this
document shall cover the technical documentation.


Documentation
-------------

.. toctree::
    :maxdepth: 1

    install
    command_line/gwgen.rst
    api/gwgen.rst
    fortran_api/index.rst


References
----------
.. [Richardson1981] Richardson, C. W.: *Stochastic simulation of daily
    precipitation, temperature, and solar radiation*, Water Resources Research,
    17, 182–190, :doi:`10.1029/WR017i001p00182`, 1981.
.. [GHCN] T. G.: Global Historical Climatology Network - Daily (GHCN-Daily),
    Version 3.22, doi:10.7289/V5D21VHZ, :doi:`10.7289/V5D21VHZ`, 2012
.. [EECRA] Hahn, C. and Warren, S.: *Extended Edited Synoptic Cloud Reports from
    Ships and Land Stations Over the Globe*, 1952-1996 (with Ship data
    updated through 2008), :doi:`10.3334/CDIAC/cli.ndp026c`, 1999


Acknowledgment
--------------
This package has been developed by Philipp Sommer and Jed Kaplan.

A special Thanks to the Shawn Koeppenhoefer for his help during the
parameterization.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

