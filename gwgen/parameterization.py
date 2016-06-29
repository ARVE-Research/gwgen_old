"""Module holding the parameterization scripts for the weather generator"""
import os
import os.path as osp
import datetime as dt
import six
import abc
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
import gwgen.utils as utils
from gwgen.utils import docstrings


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

    #: required tasks for this instance. See :meth:`set_requirements`
    _requirements = None

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
    def data_srcdir(self):
        """str. Path to the directory where the source data of the model
        is located"""
        return self.model_config['data']

    @property
    def data_dir(self):
        """str. Path to the directory were the processed data is stored"""
        ret = osp.join(self.config['expdir'], 'data')
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret

    @property
    def datafile(self):
        """str. The path to the csv file where the data is stored by the
        :meth:`Parameterizer.write2file` method and read by the
        :meth:`Parameterizer.setup_from_file`"""
        return osp.join(self.data_dir, self._datafile)

    @property
    def engine(self):
        """The sqlalchemy engine to access the database"""
        if isinstance(self._engine, six.string_types):
            from sqlalchemy import create_engine
            self._engine = create_engine(self._engine)
        return self._engine

    _logger = None

    @property
    def logger(self):
        """The logger of this organizer"""
        return self._logger or logging.getLogger('.'.join(
            [__name__, self.__class__.__name__, self.name or '']))

    @docstrings.get_sectionsf('Parameterizer')
    def __init__(self, stations, config, model_config, logger=None,
                 engine=None):
        """
        Parameters
        ----------
        stations: list
            The list of stations to process
        config: dict
            The configuration of the experiment
        model_config: dict
            The configuration of the underlying model
        logger: logging.Logger
            The logger to use for this parameterizer
        engine: sqlalchemy.Engine
            The sqlalchemy engine to use to access the database
        ``**requirements``
            Initialization keywords a defined through the :attr:`requires`
            attribute
        """
        self.stations = stations
        self.config = config
        self.model_config = model_config
        self._logger = logger
        self._engine = engine

    def set_requirements(self, **requirements):
        missing = set(self.requires).difference(requirements)
        if missing:
            raise ValueError(
                "Missing required instances for %s" % ', '.join(missing))
        self._requirements = requirements

    def _get_setup(self):
        if osp.exists(self.datafile):
            return 'file'
        elif self.engine is not None and self.engine.has_table(self.dbname):
            return 'db'
        else:
            return 'scratch'
        pass

    @docstrings.get_sectionsf('Parameterizer._setup_or_init')
    @docstrings.dedent
    def _setup_or_init(self, method=None, setup_from=None):
        """
        Method to initialize or setup the data of a parameterization task

        This method is called by :meth:`setup` and :meth:`init_task` and
        calls :meth:`setup_from_file`, :meth:`setup_from_db` and
        :meth:`setup_from_scratch` (or the corresponding `init` method)
        depending  on `setup_from`

        Parameters
        ----------
        method: { 'setup' | 'init' }
            The methods to call. If method is ``'setup'``, the (depending on
            `setup_from`), e.g. the :meth:`setup_from_scratch` is called,
            otherwise (e.g.) the :meth:`init_from_scratch` method is called
        setup_from: { 'scratch' | 'file' | 'db' }
            The method how to setup the instance either from

            scratch
                To set up the model from the raw data
            file
                Set up the model from an existing file
            db
                Set up the model from a database
            None
                If the file name of this this task exists, use this one,
                otherwise a database is provided, use this one, otherwise go
                from scratch
        """
        if self.requires and self._requirements is None:
            raise ValueError('set_requirements method has not been called!')
        setup_from = setup_from or self._get_setup()
        args = (self._engine, ) if setup_from == 'db' else ()
        return getattr(self, method + '_from_' + setup_from)(*args)

    docstrings.keep_params('Parameterizer._setup_or_init.parameters',
                           'setup_from')

    @docstrings.dedent
    def init_task(self, *args, **kwargs):
        """
        Method that is called on the I/O-Processor to initialize the
        parameterization

        Parameters
        ----------
        %(Parameterizer._setup_or_init.parameters.setup_from)s"""
        return self._setup_or_init('init', *args, **kwargs)

    @docstrings.dedent
    def setup(self, *args, **kwargs):
        """Set up the database for this parameterizer

        Parameters
        ----------
        %(Parameterizer._setup_or_init.parameters.setup_from)s
        """
        return self._setup_or_init('setup', *args, **kwargs)

    def init_from_file(self):
        """Initialize the parameterization from already stored files"""
        pass

    def init_from_db(self):
        """Initialize the parameterization from datatables already created"""
        pass

    def init_from_scratch(self):
        """Initialize the parameterization from the configuration settings"""
        pass

    def setup_from_file(self):
        """Set up the parameterizer from already stored files"""
        all_data = pd.read_csv(self.datafile, index_col='id')
        self.data = all_data.loc[self.stations]

    def setup_from_db(self):
        """Set up the parameterizer from datatables already created"""
        self.data = pd.read_sql_query(
            "SELECT * FROM %s WHERE id IN (%s)" % (
                self.dbname, ', '.join(map("'{0}'".format, self.stations))),
            self.engine, index_col='id')

    @classmethod
    def setup_from_instances(cls, instances, **kwargs):
        """Combine multiple parameterization instances into one instance"""
        base = instances[0]
        kwargs.setdefault('engine', base._engine)
        obj = cls(np.concatenate(tuple(ini.stations for ini in instances)),
                  base.config, base.model_config, **kwargs)
        obj.data = pd.concat([ini.data for ini in instances])
        return obj

    @abc.abstractmethod
    def setup_from_scratch(self):
        """Setup the data from the configuration settings"""

    @abc.abstractmethod
    def run(self):
        """Run the parameterization

        Returns
        -------
        dict
            The dictionary with the configuration settings for the namelist
        dict
            The dictionary holding additional meta information"""
        pass

    def write2db(self):
        """Write the data from this parameterizer to the database given by
        `engine`"""
        self.data.to_sql(self.dbname, self.engine, if_exists='append')

    def write2file(self):
        """Write the database to a file"""
        self.data.to_csv(self.datafile)

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

    @classmethod
    def get_requirements(cls, identifier):
        """Return the required parameterization classes for this task

        Parameters
        ----------
        identifier: str
            The :attr:`name` attribute of the :class:`Parameterizer` subclass

        Returns
        -------
        list of :class:`Parameterizer`
            A list of Parameterizer subclasses that are required for the task
            of the given `identifier`"""
        def get_requirements(parameter_cls):
            for identifier in parameter_cls.requires:
                req_cls = cls.get_parameterizer(identifier)
                ret.append(req_cls)
                get_requirements(req_cls)
        ret = []
        get_requirements(cls.get_parameterizer(identifier))
        return ret

    @classmethod
    def sort_by_requirement(cls, objects):
        """Sort the given parameterization tasks by their logical order

        Parameters
        ----------
        classes: list of :class:`Parameterizer` subclasses or instances
            The objects to sort

        Returns
        -------
        list of :class:`Parameterizer` subclasses or instances
            The same as `classes` but sorted"""
        def get_requirements(current):
            for i, parameterizer_cls in enumerate(remaining):
                if parameterizer_cls.name in current.requires:
                    get_requirements(remaining.pop(i))
                    ret.append(parameterizer_cls)
            ret.append(current)
        remaining = list(objects)
        ret = []
        while remaining:
            get_requirements(remaining.pop(0))
        return ret

    @classmethod
    @docstrings.get_sectionsf('Parameterizer._get_tasks')
    @docstrings.dedent
    def _get_tasks(cls, stations, logger=None, **kwargs):
        """
        Initaliaze the parameterization tasks

        This classmethod uses the :class:`Parameterizer` framework to
        initialize the parameterization tasks

        Parameters
        ----------
        %(Parameterizer.parameters)s
        ``**kwargs``
            Keywords can be valid identifiers of the :class:`Parameterizer`
            instances, dictionaries may be mappings for their
            :meth:`~Parameterizer.setup` method

        Returns
        -------
        list
            A list of :class:`Parameterizer` instances"""
        def init_parameterizer(task, config=None, model_config=None,
                               engine=None):
            l = logger.getChild(task.name) if logger is not None else None
            kws = kwargs[task.name]
            config = config or kws.pop('config')
            model_config = model_config or kws.pop('model_config')
            engine = engine or kws.pop('engine', None)
            return task(stations, config, model_config, logger=l,
                        engine=engine)
        if isinstance(logger, six.string_types):
            logger = logging.getLogger(logger)
        tasks = {task.name: init_parameterizer(task) for task in map(
            cls.get_parameterizer, kwargs)}
        # insert the requirements
        for key, obj in tasks.items():
            kws = kwargs[key]
            if kws.get('setup_from') is None:
                kws['setup_from'] = obj._get_setup()
            if kws['setup_from'] == 'scratch':
                for task in cls.get_requirements(key):
                    if task.name not in tasks:
                        tasks[task.name] = init_parameterizer(
                            task, task.config, task.model_config,
                            task._engine)
        # sort the tasks for their requirements
        sorted_tasks = Parameterizer.sort_by_requirement(tasks.values())
        for i, instance in enumerate(sorted_tasks):
            requirements = {ini.name: ini for ini in sorted_tasks[:i]
                            if ini.name in instance.requires}
            instance.set_requirements(**requirements)
        return sorted_tasks

    @classmethod
    @docstrings.dedent
    def initialize_parameterization(cls, stations, logger=None, **kwargs):
        """
        Initialize the parameterization

        This classmethod uses the :class:`Parameterizer` framework to
        initialize the paramterization on the I/O-processor

        Parameters
        ----------
        %(Parameterizer._get_tasks.parameters)s"""
        task_kwargs = {key: val.copy() for key, val in kwargs.items()}
        # sort the tasks for their requirements
        sorted_tasks = cls._get_tasks(stations, logger, **task_kwargs)
        for instance in sorted_tasks:
            instance.init_task(**task_kwargs.get(instance.name, {}))
        return

    @classmethod
    @docstrings.dedent
    def process_data(cls, stations, logger=None, **kwargs):
        """
        Process the given stations

        This classmethod uses the :class:`Parameterizer` framework to run the
        parameterization of the gwgen model

        Parameters
        ----------
        %(Parameterizer._get_tasks.parameters)s

        Returns
        -------
        list
            A list of :class:`Parameterizer` instances specified in ``kwargs``
            that hold the data"""
        task_kwargs = {key: val.copy() for key, val in kwargs.items()}
        # sort the tasks for their requirements
        sorted_tasks = cls._get_tasks(stations, logger, **task_kwargs)
        for instance in sorted_tasks:
            instance.setup(**task_kwargs.get(instance.name, {}))
            # the logger cannot be pickled and makes problems in
            # multiprocessing. Therefore we delete it
            del instance._logger
        return list(filter(lambda ini: ini.name in kwargs, sorted_tasks))


class DailyGHCNData(Parameterizer):
    """The parameterizer that reads in the daily data"""

    name = 'day'

    _datafile = "ghcn_daily.csv"

    dbname = 'ghcn_daily'

    summary = 'Read in the daily GHCN data'

    http_source = 'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz'

    def run(self):
        """Does nothing because this instance is only used for reading the
        data"""
        return {}, {}

    def setup_from_scratch(self):
        from gwgen.parseghcnrow import read_ghcn_file
        logger = self.logger
        stations = self.stations
        logger.debug('Reading daily ghcn data for %s stations', len(stations))
        src_dir = osp.join(self.data_srcdir, 'ghcn', 'ghcnd_all')
        logger.debug('    Data source: %s', src_dir)
        files = list(map(lambda s: osp.join(src_dir, s + '.dly'), stations))
        self.data = pd.concat(
            list(map(read_ghcn_file, files)), copy=False)

    def init_from_scratch(self):
        """Reimplemented to download the data if not existent"""
        logger = self.logger
        logger.debug('Initializing %s', self.name)
        stations = self.stations
        logger.debug('Reading data for %s stations', len(stations))
        src_dir = osp.join(self.data_srcdir, 'ghcn', 'ghcnd_all')
        logger.debug('    Expected data source: %s', src_dir)
        files = list(map(lambda s: osp.join(src_dir, s + '.dly'), stations))
        if not all(map(osp.exists, files)):
            logger.debug('    Required files are not existent in %s', src_dir)
            import tarfile
            tarfname = self.model_config.get('ghcn_src', src_dir + '.tar.gz')
            if not osp.exists(tarfname):
                logger.debug('    Downloading rawdata from %s',
                             self.http_source)
                if not osp.exists(osp.dirname(tarfname)):
                    os.makedirs(osp.dirname(tarfname))
                utils.download_file(self.http_source, tarfname)
                self.model_config['ghcn_download'] = dt.datetime.now()
                self.model_config['ghcn_src'] = tarfname
            taro = tarfile.open(tarfname, 'r|gz')
            logger.debug('    Extracting to %s', osp.dirname(src_dir))
            taro.extractall(osp.dirname(src_dir))


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
