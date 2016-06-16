"""Module holding the parameterization scripts for the weather generator"""
import os
import os.path as osp
import six
import abc
import pandas as pd
import numpy as np


def _requirement_property(requirement):
    def get_x(self):
        return self._requirements[requirement]

    return property(
        get_x, doc=requirement + " parameterization instance")


class ParameterizerMeta(abc.ABCMeta):
    """Meta class for the :class:`Parameterizer`"""

    def __new__(cls, name, bases, namespace):
        new_cls = super(ParameterizerMeta, cls).__new__(
            cls, name, bases, namespace)
        if new_cls.name:
            new_cls._registry.append(new_cls)
        for requirement in new_cls.requires:
            setattr(new_cls, requirement, _requirement_property(requirement))
        return new_cls


@six.add_metaclass(ParameterizerMeta)
class Parameterizer(object):
    """Base class for parameterization tasks

    Abstract base class that introduces the methods for one parameterization
    task. The name of the parameterization task is specified in the
    :attr:`name` attribute. You can implement the connection to other
    parameterization tasks in the :attr:`requires` attribute. The
    corresponding instances to the identifiers in the :attr:`requires`
    attribute can later be accessed through the given attribute.

    Examples
    --------
    Let's define a parameterizer that does nothing but requires another
    parameterization task named *cloud* as connection::

        >>> class CloudParameterizer(Parameterizer):
        ...     name = 'cloud'
        ...     def run(self):
        ...         pass
        ...
        >>> class DummyParameterizer(Parameterizer):
        ...     requires = ['cloud']
        ...     name = 'dummy'
        ...     def run(self):
        ...         pass
        ...
        >>> cloud = CloudParameterizer()
        >>> dummy = DummyParameterizer(cloud=cloud)
        >>> dummy.cloud is cloud
        True"""

    #: The registered parameterization classes (are set up automatically by
    #: :class:`ParameterizerMeta`)
    _registry = []

    #: list of str. identifiers of required classes for this parameterization
    #: task
    requires = []

    #: str. name of the parameterization task
    name = None

    #: pandas.DataFrame. The dataframe holding the daily data
    data = None

    #: str. The basename of the csv file where the data is stored by the
    #: :meth:`Parameterizer.write2file` method and read by the
    #: :meth:`Parameterizer.setup_from_file`
    _datafile = ""

    #: The database name to use
    dbname = ''

    #: str. summary of what this parameterizer does
    summary = ''

    @property
    def data_dir(self):
        """str. Path to the directory were the processed data is stored"""
        ret = osp.join(self.config['exp_dir'], 'data')
        if not osp.exists(ret):
            os.makedirs(ret)
        return ret

    @property
    def datafile(self):
        """str. The path to the csv file where the data is stored by the
        :meth:`Parameterizer.write2file` method and read by the
        :meth:`Parameterizer.setup_from_file`"""
        return osp.join(self.data_dir, self._datafile)

    def __init__(self, stations, config, **requirements):
        """
        Parameters
        ----------
        stations: list
            The list of stations to process
        config: dict
            The configuration of the experiment
        ``**requirements``
            Initialization keywords a defined through the :attr:`requires`
            attribute
        """
        missing = set(self.requires) - requirements
        if missing:
            raise ValueError(
                "Missing required instances for %s" % ', '.join(missing))
        self.config = config
        self._requirements = requirements

    def setup_from_file(self):
        """Set up the parameterizer from already stored files"""
        self.data = pd.read_csv(self.datafile)

    def setup_from_db(self, engine):
        """Set up the parameterizer from datatables already created"""
        self.data = pd.read_sql_table(self.dbname, engine)

    def setup_from_instances(self, instances):
        """Combine multiple parameterization instances into one instance"""
        self.data = pd.concat([ini.data for ini in instances])

    @abc.abstractmethod
    def run(self):
        """Run the parameterization

        Returns
        -------
        dict
            The dictionary with the configuration settings"""
        pass

    def write2db(self, engine):
        """Write the data from this parameterizer to the database given by
        `engine`"""
        self.data.to_sql(self.dbname, engine)

    def write2file(self):
        """Write the database to a file"""
        self.data.to_csv(self.datafile, index=False)

    @classmethod
    def get_parameterizer(cls, identifier):
        """Return the parameterization class corresponding to the `identifier`

        Parameters
        ----------
        identifier: str
            The :attr:`name` attribute of the :class:`Parameterizer` subclass

        Returns
        -------
        Parameterizer
            The parameterizer class to use"""
        return next(
            para_cls for para_cls in cls._registry[::-1]
            if para_cls.name == identifier)


class DailyGHCNData(Parameterizer):
    """The parameterizer that reads in the daily data"""

    name = 'day'

    _datafile = "daily_data.csv"

    dbname = 'daily_data'

    summary = 'Read in the daily GHCN data'

    def run(self):
        pass


class MonthlyGHCNData(Parameterizer):
    """The parameterizer that calculates the monthly summaries from the daily
    data"""

    name = 'month'

    requires = ['day']

    _datafile = "monthly_data.csv"

    dbname = 'monthly_data'

    summary = "Calculate monthly means from the daily GHCN data"


class CompleteMonthlyGHCNData(MonthlyGHCNData):
    """The parameterizer that calculates the monthly summaries from the daily
    data"""

    name = 'cmonth'

    requires = ['month']

    _datafile = "complete_monthly_data.csv"

    dbname = 'complete_monthly_data'

    summary = "Extract the complete months from the monthly data"


class CompleteDailyGHCNData(Parameterizer):
    """The parameterizer that calculates the days in complete months"""

    name = 'cday'

    requires = ['cmonth']

    _datafile = "daily_complete_months"

    dbname = 'complete_daily_data'

    summary = "Get the days of the complete months"


class PrcpDistParams(Parameterizer):
    """The parameterizer to calculate the precipitation distribution parameters
    """

    name = 'prcp'

    requires = ['cday']

    _datafile = "prcp_dist_parameters.csv"

    dbname = 'prcp_dist_params'

    summary = ('Calculate the precipitation distribution parameters of the '
               'hybrid Gamma-GP')


class MarkovChain(Parameterizer):
    """The parameterizer to calculate the Markov Chain parameters"""

    name = 'markov'

    requires = ['cday']

    _datafile = 'markov_chain_parameters.csv'

    dbname = 'markov'

    summary = ("Calculate the markov chain parameterization")
