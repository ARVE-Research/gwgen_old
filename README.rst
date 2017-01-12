GWGEN: A global weather generator for daily data
================================================


.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - tests
      - |travis| |coveralls|

.. |travis| image:: https://travis-ci.org/ARVE-Research/gwgen.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.org/ARVE-Research/gwgen

.. |coveralls| image:: https://coveralls.io/repos/github/ARVE-Research/gwgen/badge.svg?branch=master
    :alt: Coverage
    :target: https://coveralls.io/github/ARVE-Research/gwgen?branch=master

.. end-badges


Welcome! This synthesis of FORTRAN and Python is a globally applicable
weather generator inspired by the original WGEN weather generator of
[Richardson1981]_ and parameterized through a global dataset of [GHCN]_ and
[EECRA]_ data. A paper with the scientific documentation is in progress, the
`technical documentation`_ with more information is hosted on github.io.

.. _technical documentation: https://arve-research.github.io/gwgen/


Authors
-------
This package has been developped by `Philipp S. Sommer`_ and `Jed O. Kaplan`_.
We also want to acknowledge `Dr. Shawn Koppenhoefer`_ for his help in the
beginning of the project, as well as the Swiss National Fund (SNF) and the
European Research Council (ERC) for their funding.

.. _Philipp S. Sommer: https://github.com/Chilipp
.. _Jed O. Kaplan: https://github.com/jedokaplan
.. _Dr. Shawn Koppenhoefer: http://arve.unil.ch/people/shawn-koppenhoefer/

References
----------
.. [Richardson1981] Richardson, C. W.: *Stochastic simulation of daily
    precipitation, temperature, and solar radiation*, Water Resources Research,
    17, 182–190, doi:10.1029/WR017i001p00182, 1981.
.. [GHCN] T. G.: Global Historical Climatology Network - Daily (GHCN-Daily),
    Version 3.22, doi:10.7289/V5D21VHZ, doi:10.7289/V5D21VHZ, 2012
.. [EECRA] Hahn, C. and Warren, S.: *Extended Edited Synoptic Cloud Reports from
    Ships and Land Stations Over the Globe*, 1952-1996 (with Ship data
    updated through 2008), doi:10.3334/CDIAC/cli.ndp026c, 1999
