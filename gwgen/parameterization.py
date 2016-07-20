"""Module holding the parameterization scripts for the weather generator"""
import os
import glob
import os.path as osp
import datetime as dt
import six
import abc
import inspect
import subprocess as spr
from itertools import chain, product, starmap, filterfalse
import logging
import pandas as pd
import numpy as np
import calendar
import gwgen.utils as utils
from gwgen.utils import docstrings
from psyplot.compat.pycompat import OrderedDict

try:
    import copyreg as cr
except ImportError:
    import copy_reg as cr


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
        ...     def setup_from_scratch(self):
        ...         pass
        ...
        >>> class DummyParameterizer(Parameterizer):
        ...     requires = ['cloud']
        ...     name = 'dummy'
        ...     def setup_from_scratch(self):
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
    
    #: dict. Formatoptions to use when making plots with this parameterization
    #: task
    fmt = {}

    #: bool. Boolean that is True if there is a run method for this 
    #: parameterization task
    has_run = False

    @property
    def data_srcdir(self):
        """str. Path to the directory where the source data of the model
        is located"""
        return self.model_config['data']

    @property
    def data_dir(self):
        """str. Path to the directory were the processed data is stored"""
        ret = osp.join(self.config['expdir'], 'parameterization')
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
    def nc_file(self):
        """NetCDF file for the project"""
        return osp.join(self.data_dir, self.name + '.nc')
        
    @property
    def project_file(self):
        """Pickle file for the project"""
        return osp.join(self.data_dir, self.name + '.pkl')
        
    @property
    def pdf_file(self):
        """pdf file with figures the project"""
        return osp.join(self.data_dir, self.name + '.pdf')

    @property
    def engine(self):
        """The sqlalchemy engine to access the database"""
        if isinstance(self._engine, six.string_types):
            from sqlalchemy import create_engine
            return create_engine(self._engine)
        return self._engine

    @property
    def sql_dtypes(self):
        """The data types to write the data into a postgres database"""
        import sqlalchemy
        dtype={
            'station_id': sqlalchemy.INTEGER, 'tmin': sqlalchemy.REAL,
            'id': sqlalchemy.CHAR(length=11), 'prcp': sqlalchemy.REAL,
            'tmax': sqlalchemy.REAL, 'mean_cloud': sqlalchemy.REAL,
            'wet_day': sqlalchemy.SMALLINT, 'ndaymon': sqlalchemy.SMALLINT,
            'year': sqlalchemy.SMALLINT, 'month': sqlalchemy.SMALLINT,
            'day': sqlalchemy.SMALLINT}
        names = list(chain(self.data.columns, self.data.index.names))
        return {key: val for key, val in dtype.items() if key in names}

    _logger = None

    @property
    def logger(self):
        """The logger of this organizer"""
        return self._logger or logging.getLogger('.'.join(
            [__name__, self.__class__.__name__, self.name or '']))
        
    @staticmethod
    def _get_copy_reg(parameterizer):
        if parameterizer._engine is None:
            engine = None
        else:
            engine = parameterizer.engine.url 
        return parameterizer.__class__, (
            parameterizer.stations, parameterizer.config, 
            parameterizer.model_config, parameterizer.logger.name, engine)

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
        if isinstance(logger, six.string_types):
            logger = logging.getLogger(logger)
        self._logger = logger
        self._engine = engine
        # overwrite the class attribute of the formatoptions
        self.fmt = self.fmt.copy()

    def set_requirements(self, **requirements):
        missing = set(self.requires).difference(requirements)
        if missing:
            raise ValueError(
                "%s requires instances for paramterization of %s!" % (
                    self.name, ', '.join(missing)))
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

    def setup_from_file(self, **kwargs):
        """Set up the parameterizer from already stored files"""
        all_data = pd.read_csv(self.datafile, **kwargs)
        if 'id' in all_data.columns:
            all_data.set_index('id', inplace=True)
        if len(all_data.index.names) == 1:
            self.data = all_data.loc[list(self.stations)]
        else:
            i = all_data.index.names.index('id')
            self.data = all_data.sort_index().loc(axis=i)[list(self.stations)]

    def setup_from_db(self, **kwargs):
        """Set up the parameterizer from datatables already created"""
        self.data = pd.read_sql_query(
            "SELECT * FROM %s WHERE id IN (%s)" % (
                self.dbname, ', '.join(map("'{0}'".format, self.stations))),
            self.engine, **kwargs)

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

    @docstrings.get_sectionsf('Parameterizer.run', 
                              sections=['Parameters', 'Returns'])
    @docstrings.dedent
    def run(self, plot_output=None, nc_output=None, project_output=None,
            new_project=False, project=None):
        """
        Run the parameterization
        
        Parameters
        ----------
        plot_output: str
            An alternative path to use for the PDF file of the plot
        nc_output: str
            An alternative path to use for the netCDF file of the plot data
        project_output: str
            An alternative path to use for the psyplot project file of the plot
        new_project: bool
            If True, a new project will be created even if a file in
            `project_output` exists already
        project: str
            The path to a psyplot project file to use for this parameterization
        
        Returns
        -------
        dict
            The dictionary with the configuration settings for the namelist
        dict
            The dictionary holding additional meta information"""
        return {}, {}

    @classmethod    
    def _modify_parser(cls, parser):
        parser.update_arg('plot_output', short='o')
        parser.update_arg('nc_output', short='onc')
        parser.update_arg('project_output', short='op')
        parser.update_arg('new_project', short='np')
        parser.update_arg('project', short='p')
        return parser
        
    def get_run_kws(self, kwargs):
        return {key: val for key, val in kwargs.items() 
                if key in inspect.getargspec(self.run)[0]}

    def write2db(self, **kwargs):
        """Write the data from this parameterizer to the database given by
        `engine`"""
        if 'id' in self.data.columns:
            data = self.data.set_index('id')
        else:
            data = self.data
        dtype = self.sql_dtypes
        missing = set(chain(data.columns, data.index.names)).difference(dtype)
        if missing:
            self.logger.warn('No data type was specified for %s', missing)
            dtype = None
        else:
            kwargs.setdefault('dtype', dtype)
        dbname = self.dbname
        self.logger.info('Writing data to data table %s', dbname)
        data.to_sql(dbname, self.engine, if_exists='append', **kwargs)

    def write2file(self):
        """Write the database to a file"""
        datafile = self.datafile
        self.logger.info('Writing data to %s', datafile)
        self.data.to_csv(datafile)

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
            for name, parameterizer_cls in list(remaining.items()):
                if name in current.requires and name in remaining:
                    get_requirements(remaining.pop(name))
            ret.append(current)
        remaining = {task.name: task for task in objects}
        ret = []
        while remaining:
            get_requirements(remaining.pop(next(iter(remaining))))
        return ret

    @classmethod
    @docstrings.get_sectionsf('Parameterizer._get_tasks')
    @docstrings.dedent
    def _get_tasks(cls, stations, logger=None, task_kws={}):
        """
        Initaliaze the parameterization tasks

        This classmethod uses the :class:`Parameterizer` framework to
        initialize the parameterization tasks

        Parameters
        ----------
        %(Parameterizer.parameters)s
        task_kws: dict
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
            kws = task_kws.get(task.name, {})
            config = config or kws.pop('config')
            model_config = model_config or kws.pop('model_config')
            engine = engine or kws.pop('engine', None)
            return task(stations, config, model_config, logger=l,
                        engine=engine)
        if isinstance(logger, six.string_types):
            logger = logging.getLogger(logger)
        tasks = {task.name: init_parameterizer(task) for task in map(
            cls.get_parameterizer, task_kws)}
        task_kws = task_kws.copy()
        # insert the requirements
        checked_requirements = False
        while not checked_requirements:
            checked_requirements = True
            for key, task in tasks.copy().items():
                kws = task_kws[key]
                if kws.get('setup_from') is None:
                    kws['setup_from'] = task._get_setup()
                if kws['setup_from'] == 'scratch':
                    for req_task in cls.get_requirements(key):
                        if req_task.name not in tasks:
                            checked_requirements = False
                            tasks[req_task.name] = init_parameterizer(
                                req_task, task.config, task.model_config,
                                task._engine)
                            task_kws[req_task.name] = kws.copy()
                            del task_kws[req_task.name]['setup_from']
        # sort the tasks for their requirements
        sorted_tasks = list(Parameterizer.sort_by_requirement(tasks.values()))
        for i, instance in enumerate(sorted_tasks):
            if task_kws[instance.name]['setup_from'] == 'scratch':
                requirements = {ini.name: ini for ini in sorted_tasks[:i]
                                if ini.name in instance.requires}
                instance.set_requirements(**requirements)
            else:
                instance.set_requirements(
                    **{key: None for key in instance.requires})
        return sorted_tasks

    @classmethod
    @docstrings.dedent
    def initialize_parameterization(cls, stations, logger=None, task_kws={}):
        """
        Initialize the parameterization

        This classmethod uses the :class:`Parameterizer` framework to
        initialize the paramterization on the I/O-processor

        Parameters
        ----------
        %(Parameterizer._get_tasks.parameters)s"""
        task_kws = {key: val.copy() for key, val in task_kws.items()}
        # sort the tasks for their requirements
        sorted_tasks = cls._get_tasks(stations, logger, task_kws)
        for instance in sorted_tasks:
            instance.init_task(**task_kws.get(instance.name, {}))
        return

    @classmethod
    @docstrings.dedent
    def process_data(cls, stations, logger=None, task_kws={}):
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
        task_kws = {key: val.copy() for key, val in task_kws.items()}
        # sort the tasks for their requirements
        sorted_tasks = cls._get_tasks(stations, logger, task_kws)
        for instance in sorted_tasks:
            instance.setup(**task_kws.get(instance.name, {}))
            # the logger cannot be pickled and makes problems in
            # multiprocessing. Therefore we delete it
            try:
                del instance._logger
            except AttributeError:
                pass
        return list(filter(lambda ini: ini.name in task_kws, sorted_tasks))


class DailyGHCNData(Parameterizer):
    """The parameterizer that reads in the daily data"""

    name = 'day'

    _datafile = "ghcn_daily.csv"

    dbname = 'ghcn_daily'

    summary = 'Read in the daily GHCN data'

    http_source = 'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz'
    
    @property
    def data_srcdir(self):
        return osp.join(super(HourlyCloud, self).data_srcdir, 
                        'ghcn', 'ghcnd_all')

    @property
    def sql_dtypes(self):
        import sqlalchemy
        ret = super(DailyGHCNData, self).sql_dtypes
        flags = ['tmax_m', 'prcp_s', 'tmax_q', 'prcp_m', 'tmin_m', 'tmax_s',
                 'tmin_s', 'prcp_q', 'tmin_q']
        ret.update({flag: sqlalchemy.CHAR(length=1) for flag in flags})
        return ret

    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day']
        kwargs['dtype'] = {
            'prcp': np.float64,
            'prcp_m': object,
            'prcp_q': object,
            'prcp_s': object,
            'tmax': np.float64,
            'tmax_m': object,
            'tmax_q': object,
            'tmax_s': object,
            'tmin': np.float64,
            'tmin_m': object,
            'tmin_q': object,
            'tmin_s': object}
        return super(DailyGHCNData, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day']
        return super(DailyGHCNData, self).setup_from_db(*args, **kwargs)

    def setup_from_scratch(self):
        from gwgen.parseghcnrow import read_ghcn_file
        logger = self.logger
        stations = self.stations
        logger.debug('Reading daily ghcn data for %s stations', len(stations))
        src_dir = self.data_srcdir
        logger.debug('    Data source: %s', src_dir)
        files = list(map(lambda s: osp.join(src_dir, s + '.dly'), stations))
        self.data = pd.concat(
            list(map(read_ghcn_file, files)), copy=False).set_index(
                ['id', 'year', 'month', 'day'])
        self.logger.debug('Done.')

    def init_from_scratch(self):
        """Reimplemented to download the data if not existent"""
        logger = self.logger
        logger.debug('Initializing %s', self.name)
        stations = self.stations
        logger.debug('Reading data for %s stations', len(stations))
        src_dir = self.data_srcdir
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

    _datafile = "ghcn_monthly.csv"

    dbname = 'ghcn_monthly'

    summary = "Calculate monthly means from the daily GHCN data"

    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month']
        return super(MonthlyGHCNData, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month']
        return super(MonthlyGHCNData, self).setup_from_db(*args, **kwargs)

    @staticmethod
    def monthly_summary(df):
        n = calendar.monthrange(df.index[0][1], df.index[0][2])[1]
        return pd.DataFrame.from_dict(
            {'tmin': [df.tmin.mean()], 'tmax': [df.tmax.mean()],
             'trange': [(df.tmax - df.tmin).mean()],
             'prcp': [df.prcp.sum()], 'tmin_abs': [df.tmin.min()],
             'tmax_abs': [df.tmax.max()], 'prcpmax': [df.prcp.max()],
             'tmin_complete': [df.tmin.count() == n],
             'tmax_complete': [df.tmax.count() == n],
             'prcp_complete': [df.prcp.count() == n]})

    def setup_from_scratch(self):
        data = self.day.data.groupby(level=['id', 'year', 'month']).apply(
            self.monthly_summary)
        data.index = data.index.droplevel(-1)
        self.data = data


class CompleteMonthlyGHCNData(MonthlyGHCNData):
    """The parameterizer that calculates the monthly summaries from the daily
    data"""

    name = 'cmonth'

    requires = ['month']

    _datafile = "complete_ghcn_monthly.csv"

    dbname = 'complete_ghcn_monthly'

    summary = "Extract the complete months from the monthly data"

    def setup_from_scratch(self):
        all_months = self.month.data
        self.data = all_months[all_months.prcp_complete &
                               all_months.tmin_complete &
                               all_months.tmax_complete]


class CompleteDailyGHCNData(DailyGHCNData):
    """The parameterizer that calculates the days in complete months"""

    name = 'cday'

    requires = ['day', 'month']

    _datafile = "complete_ghcn_daily.csv"

    dbname = 'complete_ghcn_daily'

    summary = "Get the days of the complete months"
    
    def init_from_scratch(self):
        pass

    def setup_from_scratch(self):
        monthly = self.month.data
        self.data = self.day.data.reset_index().merge(
            monthly[monthly.prcp_complete &
                    monthly.tmin_complete &
                    monthly.tmax_complete][[]].reset_index(),
            how='inner', on=['id', 'year', 'month'], copy=False).set_index(
                ['id', 'year', 'month', 'day'])


class PrcpDistParams(Parameterizer):
    """The parameterizer to calculate the precipitation distribution parameters
    """

    name = 'prcp'

    requires = ['cday']

    _datafile = "prcp_dist_parameters.csv"

    dbname = 'prcp_dist_params'

    summary = ('Calculate the precipitation distribution parameters of the '
               'hybrid Gamma-GP')
    
    has_run = True
    
    #: default formatoptions for the 
    #: :class:`psyplot.plotter.linreg.DensityRegPlotter` plotter
    fmt = kwargs = dict(
        legend={'loc': 'upper left'},
        cmap='w_Reds',
        precision=0.1,
        xlabel='{desc}',
        ylabel='{desc}',
        xrange=(0, ['rounded', 95]),
        yrange=(0, ['rounded', 95]),
        fix=0,
        legendlabels=['$\\theta$ = %(slope)1.4f * $\\bar{{p}}_d$'],
        bounds=['minmax', 11, 0, 99],
        cbar='',
        bins=100,
        )
    
    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'month']
        return super(PrcpDistParams, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'month']
        return super(PrcpDistParams, self).setup_from_db(*args, **kwargs)
    
    @staticmethod
    def prcp_dist_params(
            df, threshs=np.array([5, 7.5, 10, 12.5, 15, 17.5, 20])):
        from scipy import stats
        vals = df.prcp.values[~np.isnan(df.prcp.values)]
        N = len(threshs)
        n = len(vals) * N
        vals = vals[vals > 0]
        ngamma = len(vals)
        ngp = [np.nan] * N
        gshape = np.nan
        gscale = np.nan
        pshape = [np.nan] * N
        pscale = [np.nan] * N
        pscale_orig = [np.nan] * N
        if ngamma > 10:
            # fit the gamma curve. We fix the (unnecessary) location parameter to
            # improve the result (see http://stackoverflow.com/questions/16963415/why-does-the-gamma-distribution-in-scipy-have-three-parameters)
            gshape, _, gscale = stats.gamma.fit(vals, floc=0)
            for i, thresh in enumerate(threshs):
                arr = vals[vals >= thresh]
                ngp[i] = len(arr)
                if ngp[i] > 10:
                    pshape[i], _, pscale_orig[i] = stats.genpareto.fit(
                        arr, floc=thresh)
                    # find the crossover point where the gamma and pareto
                    # distributions should match
                    # this follows Neykov et al. (Nat. Hazards Earth Syst. Sci.,
                    # 14, 2321-2335, 2014) bottom of page 2330 (left column)
                    pscale[i] = (1 - stats.gamma.cdf(
                        thresh, gshape, scale=gscale))/stats.gamma.pdf(
                            thresh, gshape, scale=gscale)
        return pd.DataFrame.from_dict(
            {'n': np.repeat(n, N), 'ngamma': np.repeat(ngamma, N),
             'mean_wet': np.repeat(vals.mean(), N),
             'ngp': ngp, 'thresh': threshs, 'gshape': np.repeat(gshape, N),
             'gscale': np.repeat(gscale, N), 'pshape': pshape,
             'pscale': pscale, 'pscale_orig': pscale_orig}).set_index('thresh')
        
    def setup_from_scratch(self):
        self.logger.debug('Calculating precipitation parameters.')
        df = self.cday.data
        self.data = df.groupby(level=['id', 'month']).apply(
            self.prcp_dist_params)
        self.logger.debug('Done.')
        
    @docstrings.dedent
    def run(self, plot_output=None, nc_output=None, project_output=None,
            new_project=False, project=None):
        """
        Run the parameterization
        
        Parameters
        ----------
        %(Parameterizer.run.parameters)s
        
        Returns
        -------
        %(Parameterizer.run.returns)s
        """
        self.logger.info('Calculating %s parameterization', self.name)
        ret_nml, ret_info, pdf = self._run_gscale(
            False, plot_output, nc_output, project_output, new_project,
            project)
        nml, info = self._run_gpshape(pdf)
        ret_nml['weathergen'].update(nml)
        ret_info.update(info)
        pdf.close()
        return ret_nml, ret_info
        
    def _run_gscale(self, close_pdf=True, plot_output=None, nc_output=None, 
                    project_output=None, new_project=False, project=None):
        import psyplot.project as psy
        import xarray as xr
        ret_nml = {}
        ret_info = OrderedDict()
        ret_nml['weathergen'] = nml = {}
        # ---- open dataset
        ds = xr.Dataset.from_dataframe(
            self.data.set_index('mean_wet')[['gscale']])
        ds.mean_wet.attrs['long_name'] = 'Mean precip. on wet days'
        ds.mean_wet.attrs['units'] = 'mm'
        ds.gscale.attrs['long_name'] = 'Gamma scale parameter'
        ds.gscale.attrs['units'] = 'mm'
        # ---- file names
        nc_output = nc_output or self.nc_file
        plot_output = plot_output or self.pdf_file
        project_output = project_output or self.project_file
        ret_info['nc_file'] = nc_output
        ret_info['plot_file'] = plot_output
        ret_info['project_file'] = project_output
        # ---- create project
        if not new_project and osp.exists(project or project_output):
            sp = psy.Project.load_project(
                project or project_output, datasets=[ds])
        else:
            sp = psy.plot.densityreg(ds, name='gscale', fmt=self.fmt)
        for key in ['rsquared', 'slope', 'intercept']:
            ret_info[key] = float(sp.plotters[0].plot_data[1].attrs[key])
        nml['g_scale_coeff'] = float(
            sp.plotters[0].plot_data[1].attrs['slope'])
        # ---- save data and project
        pdf = sp.export(plot_output, tight=True, close_pdf=close_pdf)
        sp.save_project(project_output, use_rel_paths=True, 
                        paths=[nc_output])
        sp.close(True, True)
        return ret_nml, ret_info, pdf
        
    def _run_gpshape(self, pdf=None):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_style('darkgrid')
        
        df = self.data
        
        fig = plt.figure(figsize=(12, 2.5))
        fig.subplots_adjust(hspace=0)
        ax2 = plt.subplot2grid((4, 2), (0, 1))
        ax3 = plt.subplot2grid((4, 2), (1, 1), rowspan=3, sharex=ax2)
        pshape = df.pshape[df.ngp > 100]
        sns.boxplot(pshape, ax=ax2, whis=[1, 99], showmeans=True, 
                    meanline=True)
        sns.distplot(pshape, hist=True, kde=True, ax=ax3)
        ax3.set_xlim(*np.percentile(pshape, [0.1, 99.9]))
        ax3.set_xlabel('Generalized Pareto Shape parameter')
        ax3.set_ylabel('Counts')
        
        median = float(np.round(pshape.median(), 4))
        mean = float(np.round(pshape.mean(), 4))
        std = float(np.round(pshape.std(), 4))
        
        median_line = next(
            l for l in ax2.lines if np.all(np.round(l.get_xdata(), 4) == median))
        mean_line = next(
            l for l in ax2.lines if np.all(np.round(l.get_xdata(), 4) == mean))
        ax3.legend(
            (median_line, mean_line),
            ('median = %1.4f' % median, 'mean = %1.4f' % mean), loc='center',
            bbox_to_anchor=[0.7, 0.2], bbox_transform=ax3.transAxes)
        if pdf:
            pdf.savefig(fig, bbox_inches='tight')
        else:
            plt.savefig(self.pdf_file, bbox_inches='tight')
        
        nml = dict(gp_shape=mean)
        
        info = dict(gpshape_mean=mean, gpshape_median=median, std=std)
        
        return nml, info
        

class MarkovChain(Parameterizer):
    """The parameterizer to calculate the Markov Chain parameters"""

    name = 'markov'

    requires = ['cday']

    _datafile = 'markov_chain_parameters.csv'

    dbname = 'markov'

    summary = "Calculate the markov chain parameterization"
    
    has_run = True
    
    fmt = kwargs = dict(
        legend={'loc': 'upper left'},
        cmap='w_Reds',
        ylabel='%(long_name)s',
        xlim=(0, 1),
        ylim=(0, 1),
        fix=0,
        bins=100,
        bounds=['minmax', 11, 0, 99],
        cbar='',
        ci=None,
        legendlabels=['$%(symbol)s$ = %(slope)1.4f * %(xname)s'],
        )
    
    
    
    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'month']
        return super(MarkovChain, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'month']
        return super(MarkovChain, self).setup_from_db(*args, **kwargs)
    
    @classmethod
    def calc_ndays(cls, df):
        if not len(df):
            return pd.DataFrame([[np.nan] * 9], columns=[
                'n', 'nwet', 'ndry', 'np11', 'np01',  'np001', 'np001_denom',
                'np101', 'np101_denom'], dtype=int)
        vals = df.prcp.values
        n = len(vals)
        nwet = (vals[:-1] > 0.0).sum()   #: number of wet days
        ndry = (vals[:-1] == 0.0).sum()  #: number of dry days
        # ---------------------------------------------------
        # PWW = Prob(Wet then Wet) = P11
        #     = wetwet / wetwet + wetdry
        #     = wetwet / nwet
        np11 = ((vals[:-1] > 0.0) & (vals[1:] > 0.0)).sum()
        # ---------------------------------------------------
        # PWD = Prob(Dry then Wet) = P01
        #     = drywet / drywet + drydry
        #     = wetwet / ndry
        np01 = ((vals[:-1] == 0.0) & (vals[1:] > 0.0)).sum()
        # ---------------------------------------------------
        # PWDD = Prob(Dry Dry Wet) = P001
        #      = drydryWET / (drydryWET + drydryDRY)
        np001 = ((vals[:-2] == 0.0) & (vals[1:-1] == 0.0) &
                 (vals[2:] > 0.0)).sum()
        np001_denom = np001 + ((vals[:-2] == 0.0) & (vals[1:-1] == 0.0) &
                               (vals[2:] == 0.0)).sum()
        # ---------------------------------------------------
        # PWDW = Prob(Wet Dry Wet) = P101
        #      = wetdryWET / (wetdryWET + wetdryDRY)
        np101 = ((vals[:-2] > 0.0) & (vals[1:-1] == 0.0) &
                 (vals[2:] > 0.0)).sum()
        np101_denom = np101 + ((vals[:-2] > 0.0) & (vals[1:-1] == 0.0) &
                               (vals[2:] == 0.0)).sum()
        return pd.DataFrame(
            [[n, nwet, ndry, np11, np01, np001, np001_denom, np101, np101_denom]],
            columns=['n', 'nwet', 'ndry', 'np11', 'np01', 'np001', 'np001_denom',
                     'np101', 'np101_denom'])
    @classmethod  
    def calculate_probabilities(cls, df):
        """Calculate the transition probabilities for one month across multiple
        years"""
        # we group here for each month because we do not want to treat each 
        # month separately
        g = df.groupby(level=['year', 'month'])
        if g.ngroups > 10:
            dfs = g.apply(cls.calc_ndays).sum()
            return pd.DataFrame.from_dict(
                {'p11': [dfs.np11 / dfs.nwet if dfs.nwet > 0 else 0],
                 'p01': [dfs.np01 / dfs.ndry if dfs.ndry > 0 else 0],
                 'p001': [dfs.np001 / dfs.np001_denom 
                          if dfs.np001_denom > 0 else 0],
                 'p101': [dfs.np101 / dfs.np101_denom 
                          if dfs.np101_denom > 0 else 0],
                 'wetf': [dfs.nwet / dfs.n if dfs.n > 0 else 0]})
        else:
            return pd.DataFrame.from_dict(
                {'p11': [], 'p01': [], 'p001': [], 'p101': [], 'wetf': []})

    def setup_from_scratch(self):
        self.logger.debug('Calculating markov chain parameters')
        df = self.cday.data
        data = df.groupby(level=['id', 'month']).apply(
            self.calculate_probabilities)
        data.index = data.index.droplevel(-1)
        self.data = data
        self.logger.debug('Done.')
        
    @docstrings.dedent
    def run(self, plot_output=None, nc_output=None, project_output=None,
            new_project=False, project=None):
        """
        Run the parameterization
        
        Parameters
        ----------
        %(Parameterizer.run.parameters)s
        
        Returns
        -------
        %(Parameterizer.run.returns)s
        """
        import seaborn as sns
        import xarray as xr
        import psyplot.project as psy
        import matplotlib.pyplot as plt
        
        self.logger.info('Calculating %s parameterization', self.name)
        
        ret_nml = {}
        ret_info = OrderedDict()
        ret_nml['weathergen'] = nml = {}

        # ---- file names
        nc_output = nc_output or self.nc_file
        plot_output = plot_output or self.pdf_file
        project_output = project_output or self.project_file
        ret_info['nc_file'] = nc_output
        ret_info['plot_file'] = plot_output
        ret_info['project_file'] = project_output
        
        ds = xr.Dataset.from_dataframe(self.data.set_index('wetf'))
        ds.wetf.attrs['long_name'] = 'Fraction of wet days'
        ds.p11.attrs['long_name'] = 'Prob. Wet then Wet'
        ds.p101.attrs['long_name'] = 'Prob. Wet then Dry then Wet'
        ds.p001.attrs['long_name'] = 'Prob. Dry then Dry then Wet'
        ds.p11.attrs['symbol'] = 'p_{11}'
        ds.p101.attrs['symbol'] = 'p_{101}'
        ds.p001.attrs['symbol'] = 'p_{001}'
        
        sns.set_style("white")
        # ---- create project
        if not new_project and osp.exists(project or project_output):
            sp = psy.Project.load_project(project or project_output, 
                                          datasets=[ds])
        else:
            new_project = True
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            axes = axes.ravel()
            sp = psy.plot.densityreg(
                ds, name=['p11', 'p101', 'p001'], fmt=self.fmt, ax=axes.ravel(),
                share='xlim')
            sp(name='p11').update(
                fix=[(1., 1.)], legendlabels=[
                    '$%(symbol)s$ = %(intercept)1.4f + '
                    '%(slope)1.4f * %(xname)s'])
            sp(ax=axes[-1]).update(xlabel='%(long_name)s',)
       
        for key in ['rsquared', 'slope', 'intercept']:
            ret_info[key] = float(sp.plotters[0].plot_data[1].attrs[key])
        for plotter in sp.plotters:
            name = plotter.data.name
            nml[name + '_1'] = float(plotter.plot_data[1].attrs.get(
                'intercept', 0))
            nml[name + '_2'] = float(plotter.plot_data[1].attrs.get('slope'))
        # ---- save data and project
        sp.export(plot_output, tight=True)
        sp.save_project(project_output, 
                        use_rel_paths=True, paths=[nc_output])
        sp.close(True, True)        
        
        return ret_nml, ret_info
        
        
class TemperatureParameterizer(Parameterizer):
    """Parameterizer to correlate the monthly mean and standard deviation on
    wet and dry days with the montly mean"""
    
    name = 'temp'
    
    summary = 'Temperature mean correlations'
    
    requires =['cday']

    _datafile = 'temperature.csv'
    
    dbname = 'temperature'

    has_run = True
    
    fmt = dict(
        legend={'loc': 'upper left'},
        cmap='w_Reds',
        precision=0.1,
        xrange=(0, ['rounded', 95]),
        yrange=(0, ['rounded', 95]),
        legendlabels=[
            '$%(symbol)s$ = %(intercept)1.4f + %(slope)1.4f * $%(xsymbol)s$'],
        bounds=['minmax', 11, 0, 99],
        cbar='',
        bins=100,
        xlabel='on %(state)s days'
        )
    
    @property
    def ds(self):
        """The dataframe of this parameterization task converted to a dataset
        """
        import xarray as xr
        ds = xr.Dataset.from_dataframe(self.data[[
            col for col in self.data.columns if (
                col.startswith('tmin') or col.startswith('tmax'))]].set_index(
                    'tmin'))
        ds.set_coords('tmax', inplace=True)
        tmax_variables = [v for v in ds.variables if v.startswith('tmax')]
        ds_tmax = xr.Dataset.from_dataframe(self.data.set_index('tmax'))
        ds.tmin.attrs['long_name'] = 'min. temperature'
        ds.tmax.attrs['long_name'] = 'max. temperature'
        for v in tmax_variables:
            ds[v] = ds_tmax.variables[v]
            ds[v].attrs['units'] = 'degC'
            ds[v.replace('max', 'min')].attrs['units'] = 'degC'
        for v, state in product(('tmin', 'tmax'), ('wet', 'dry')):
            vname =  v + '_' + state
            std = v + 'stddev_' + state
            coord = 'c_' + vname
            ds[vname].attrs['long_name'] = 'mean %s. temperature' % (v[1:])
            ds[coord] = ds[vname].rename({v: coord})
            ds[std] = ds[std].rename({v: coord}).variable
            ds[std].attrs['long_name'] = (
                'std. dev. of %s. temperature' % (v[1:]))
            for name in [vname, coord, std, v, v + 'stddev']:
                ds[name].attrs['state'] = state if name != v else 'all'
                ds[name].attrs['symbol'] = 't_\mathrm{{%s%s%s}}' % (
                    v[1:], ', sd' if 'stddev' in name else '', 
                    (', ' + state) if name != v else '')
        return ds
    
    @staticmethod
    def calc_monthly_props(df):
        """
        Calculate the statistics for one single month in one year
        """
        prcp_vals = df.prcp.values
        wet = prcp_vals > 0.0
        dry = prcp_vals == 0
        arr_tmin = df.tmin.values
        arr_tmax = df.tmax.values
        arr_tmin_wet = arr_tmin[wet]
        arr_tmin_dry = arr_tmin[dry]
        arr_tmax_wet = arr_tmax[wet]
        arr_tmax_dry = arr_tmax[dry]
        # prcp values
        d = {
            # wet values
            'tmin_wet': arr_tmin_wet.mean(),
            'tmax_wet': arr_tmax_wet.mean(),
            'tminstddev_wet': arr_tmin_wet.std(),
            'tmaxstddev_wet': arr_tmax_wet.std(),
            'trange_wet': (arr_tmax_wet - arr_tmax_wet).mean(),
            'trangestddev_wet': (arr_tmax_wet - arr_tmax_wet).std(),
            # dry values
            'tmin_dry': arr_tmin_dry.mean(),
            'tmax_dry': arr_tmax_dry.mean(),
            'tminstddev_dry': arr_tmin_dry.std(),
            'tmaxstddev_dry': arr_tmax_dry.std(),
            'trange_dry': (arr_tmax_dry - arr_tmax_dry).mean(),
            'trangestddev_dry': (arr_tmax_dry - arr_tmax_dry).std(),
            # general mean
            'tmin': arr_tmin.mean(),
            'tmax': arr_tmax.mean(),
            'tminstddev': arr_tmin.std(),
            'tmaxstddev': arr_tmax.std(),
            'trange': (arr_tmin - arr_tmax).mean(),
            'trangestddev': (arr_tmin - arr_tmax).std(),
            't': ((arr_tmin + arr_tmax) * 0.5).mean(),
            'tstddev': ((arr_tmin + arr_tmax) * 0.5).std()}
        d['prcp_wet'] = am = prcp_vals[wet].mean()  # arithmetic mean
        gm = np.exp(np.log(prcp_vals[wet]).mean())  # geometric mean
        fields = am != gm
        d['alpha'] = (
            0.5000876 / np.log(am[fields] / gm[fields]) + 0.16488552 -
            0.0544274 * np.log(am[fields] / gm[fields]))
        d['beta'] = am / d['alpha']
        return pd.DataFrame.from_dict(d)
        
    @classmethod
    def calculate_probabilities(cls, df):
        """Calculate the statistics for one month across multiple years"""
        # we group here for each month because we do not want to treat each
        # month separately
        g = df.groupby(level=['year'])
        return g.apply(cls.calc_monthly_props).mean()
        
    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'month']
        return super(TemperatureParameterizer, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'month']
        return super(TemperatureParameterizer, self).setup_from_db(*args, **kwargs)

    def setup_from_scratch(self):
        self.data = self.cday.data.groupby(level=['id', 'month']).apply(
            self.calculate_probabilities)
    
    @docstrings.dedent
    def run(self, plot_output=None, nc_output=None, project_output=None,
            new_project=False, project=None):
        """
        Run the parameterization
        
        Parameters
        ----------
        %(Parameterizer.run.parameters)s
        
        Returns
        -------
        %(Parameterizer.run.returns)s
        """
        import seaborn as sns
        import psyplot.project as psy
        import matplotlib.pyplot as plt
        self.logger.info('Calculating %s parameterization', self.name)
        
        sns.set_style("white")
        
        ds = self.ds
        
        ret_nml = {}
        ret_info = OrderedDict()
        ret_nml['weathergen'] = nml = {}
        
        # ---- file names
        nc_output = nc_output or self.nc_file
        plot_output = plot_output or self.pdf_file
        project_output = project_output or self.project_file
        ret_info['nc_file'] = nc_output
        ret_info['plot_file'] = plot_output
        ret_info['project_file'] = project_output

        sns.set_style("white")
        variables = ['tmin', 'tmax']
        states = ['wet' ,'dry']
        types = ['', 'stddev']
        # ---- create project
        if not new_project and osp.exists(project or project_output):
            sp = psy.Project.load_project(project or project_output, 
                                          datasets=[ds])
        else:
            axes = np.concatenate([
                plt.subplots(1, 2, figsize=(12, 4))[1] for _ in range(4)])
            for fig in set(ax.get_figure() for ax in axes):
                fig.subplots_adjust(bottom=0.25)
            middle = (
                axes[0].get_position().x0 + axes[1].get_position().x1) / 2.
            axes = iter(axes)
            for v, t in product(variables, types):
                psy.plot.densityreg(
                    ds, name='%s%s_wet' % (v, t), ax=next(axes), 
                    ylabel='%(long_name)s\non %(state)s days',
                    text=[(middle, 0.03, '%(long_name)s', 'fig', dict(
                         weight='bold', ha='center'))], fmt=self.fmt)
                psy.plot.densityreg(
                    ds, name='%s%s_dry' % (v, t), ax=next(axes),
                    ylabel='on %(state)s days', fmt=self.fmt)
            sp = psy.gcp(True)[:]
        
        for v, t, state, key in product(
                variables, types, states, ['rsquared', 'slope', 'intercept']):
            vname = '%s%s_%s' % (v, t, state)
            nml_name = v + ('_sd' if t else '') + '_' + state[0]
            ret_info[vname] = vinfo = {}
            vinfo[key] = float(sp.plotters[0].plot_data[1].attrs[key])
            plotter = sp(name=vname).plotters[0]
            nml[nml_name + '1'] = float(
                plotter.plot_data[1].attrs.get('intercept', 0))
            nml[nml_name + '2'] = float(
                plotter.plot_data[1].attrs.get('slope'))
        
        
        # ---- save data and project
        sp.export(plot_output, tight=True)
        sp.save_project(project_output, 
                        use_rel_paths=True, paths=[nc_output])
        sp.close(True, True)  
        
        return ret_nml, ret_info
        
        
class CloudParameterizerBase(Parameterizer):
    """Abstract base class for cloud parameterizers"""
    
    @property
    def stations(self):
        return self._stations
        
    @stations.setter
    def stations(self, stations):
        Norig = len(stations)
        df_map = self.eecra_ghcn_map().loc[stations].dropna()
        self._stations = df_map.index.values
        self.eecra_stations = df_map.station_id.values
        self.logger.debug('Using %i cloud stations in the %i given stations',
                          len(df_map), Norig)
    
    @staticmethod
    def eecra_ghcn_map():
        """Get a dataframe mapping from GHCN id to EECRA station_id"""
        cls = CloudParameterizerBase
        try:
            return cls._eecra_ghcn_map
        except AttributeError:
            cls._eecra_ghcn_map = pd.read_csv(osp.join(
                utils.get_module_path(inspect.getmodule(cls)), 'data', 
                'eecra_ghcn_map.csv'), index_col='id')
        return cls._eecra_ghcn_map
        
    @classmethod
    def filter_stations(cls, stations):
        """Get the GHCN stations that are also in the EECRA dataset
        
        Parameters
        ----------
        stations: np.ndarray
            A string array with stations to use
            
        Returns
        -------
        np.ndarray
            The ids in `stations` that can be mapped to the eecra dataset"""
        return cls.eecra_ghcn_map().loc[stations].dropna().index.values
        
class HourlyCloud(CloudParameterizerBase):
    """Parameterizer that loads the hourly cloud data from the EECRA database
    """
    
    name = 'hourly_cloud'
    
    summary = 'Hourly cloud data'
    
    _datafile = 'hourly_cloud.csv'
    
    dbname = 'hourly_cloud'
    
    urls = {
        ((1971, 1), (1977, 4)): 'http://cdiac.ornl.gov/ftp/ndp026c/land_197101_197704/',
        ((1977, 5), (1982, 10)): 'http://cdiac.ornl.gov/ftp/ndp026c/land_197705_198210/',
        ((1982, 11), (1987, 6)): 'http://cdiac.ornl.gov/ftp/ndp026c/land_198211_198706/',
        ((1987, 7), (1992, 2)): 'http://cdiac.ornl.gov/ftp/ndp026c/land_198707_199202/',
        ((1992, 3), (1996, 12)): 'http://cdiac.ornl.gov/ftp/ndp026c/land_199203_199612/',
        ((1997, 1), (2009, 12)): 'http://cdiac.ornl.gov/ftp/ndp026c/land_199701_200912/'}
        
    mon_map = dict(zip(
        range(1, 13), 
        "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()))
    
    _continue = False
    
    @property
    def data_srcdir(self):
        return osp.join(super(HourlyCloud, self).data_srcdir, 'eecra')
    
    @property
    def src_files(self):
        src_dir = self.data_srcdir
        return {yrmon: osp.join(src_dir, self.eecra_fname(*yrmon)) 
                for yrmon in product(range(1971, 2010), range(1, 13))}

    @classmethod 
    @docstrings.get_sectionsf('HourlyCloud.eecra_fname')
    @docstrings.dedent
    def eecra_fname(cls, year, mon, ext='.csv'):
        """The the name of the eecra file
        
        Parameters
        ---------
        year: int
            The year of the data
        month: int
            The integer of the month between 1 and 12"""
        c_mon = cls.mon_map[mon]
        c_yr = str(year)
        return c_mon + c_yr[-2:] + 'L' + ext

    @classmethod
    @docstrings.dedent
    def get_eecra_url(cls, year, mon):
        """
        Get the download path for the file for a specific year and month
        
        Parameters
        ----------
        %(HourlyCloud.eecra_fname.parameters)s
        """
        for (d0, d1), url in cls.urls.items():
            if (year, mon) >= d0 and (year, mon) <= d1:
                return url + cls.eecra_fname(year, mon, '.Z')
                
    def _parse_files(self, q):
        """Worker method to parse the files in the given queue q"""
        from gwgen.parse_eecra import parse_file
        while self._continue:
            yrmon, uncompressed_fname = q.get()
            parse_file(uncompressed_fname, yrmon[0]).to_csv(
                uncompressed_fname + '.csv', index=False)
            os.remove(uncompressed_fname)
            q.task_done()
            
    def _download_worker(self, qin, qout):
        while self._continue:
            yrmon, fname = qin.get()
            utils.download_file(self.get_eecra_url(*yrmon), fname)
            spr.call(['gzip', '-d', fname])
            qout.put((yrmon, osp.splitext(fname)[0]))
            qin.task_done()
        
    def init_from_scratch(self):
        """Reimplemented to download the data if not existent"""
        logger = self.logger
        logger.debug('Initializing %s', self.name)
        stations = self.stations
        logger.debug('Reading data for %s stations', len(stations))
        src_dir = self.data_srcdir
        if not osp.isdir(src_dir):
            os.makedirs(src_dir)
        src_files = self.src_files
        logger.debug('    Expected data source: %s', src_dir)
        src_files = self.src_files
        existing = os.listdir(src_dir)
        missing = {yrmon: fname for yrmon, fname in six.iteritems(src_files) 
                   if osp.basename(fname) not in existing}
        logger.debug('%i files are missing', len(missing))
        if missing:
            from threading import Thread
            try:
                from queue import Queue
            except ImportError:
                from Queue import Queue
            self._continue = True  # makes sure the threads are running
            q = Queue()
            download_q = Queue()
            threads = [Thread(
                target=self._parse_files, args=(q, )) for _ in range(20)]
            threads.append(Thread(
                    target=self._download_worker, args=(download_q, q)))
            for thread in threads:
                thread.setDaemon(True)
            for thread in threads:
                thread.start()
        for yrmon, fname in six.iteritems(missing):
            compressed_fname = osp.splitext(fname)[0] + '.Z'
            uncompressed_fname = osp.splitext(fname)[0]
            if not osp.exists(uncompressed_fname):
                if not osp.exists(compressed_fname):
                    download_q.put((yrmon, compressed_fname))
                else:
                    spr.call(['gzip', '-d', compressed_fname])
                    q.put((yrmon, uncompressed_fname))
            else:
                q.put((yrmon, uncompressed_fname))
        if missing:
            download_q.join()
            q.join()
            self._continue = False  # stops the threads
            logger.debug('Done')
                
    def get_data_from_files(self, files):
        def save_loc(fname):
            try:
                return pd.read_csv(fname, index_col='station_id').loc[
                    station_ids]
            except KeyError:
                return pd.DataFrame()
        station_ids = self.eecra_stations
        self.logger.debug('Extracting data for %i stations from %i files',
                          len(station_ids), len(files))
        ret = pd.concat(
            list(map(save_loc, files)), ignore_index=False, copy=False)
        ret.set_index(['id', 'year', 'month', 'day', 'hour'], 
                      inplace=True).sort_index(inplace=True)
        return ret
            
    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day', 'hour']
        return super(MonthlyGHCNData, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day', 'hour']
        return super(MonthlyGHCNData, self).setup_from_db(*args, **kwargs)
    
    def setup_from_scratch(self):
        """Set up the data"""
        files = self.src_files
        self.data = self.get_data_from_files(files.values())
        

class DailyCloud(CloudParameterizerBase):
    """Parameterizer to calculate the daily cloud values from hourly cloud data
    """
    
    name = 'daily_cloud'
    
    summary = 'Calculate the daily cloud values from hourly cloud data'
    
    _datafile = 'daily_cloud.csv'
    
    dbname = 'daily_cloud'
    
    requires = ['hourly_cloud']

    @staticmethod
    def calculate_daily(df):
        return pd.DataFrame.from_dict(OrderedDict([
            ('wet_day', [(
                (df.ww >= 50 and df.ww <= 75) or
                df.ww == 75 or
                df.ww == 77 or
                df.ww == 79 or
                (df.ww >= 80 and df.ww <= 99)).any()]),
            ('tmin', [df.AT.min()]),
            ('tmax', [df.AT.max()]),
            ('mean_cloud', [df.N.mean() / 8.]),
            ('wind', [df.WS.mean()])
            ]))
            
    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day']
        return super(MonthlyGHCNData, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month', 'day']
        return super(MonthlyGHCNData, self).setup_from_db(*args, **kwargs)
    
    def setup_from_scratch(self):
        df = self.hourly_cloud.data
        data = df.groupby(level=['id', 'year', 'month', 'day']).apply(
            self.calculate_daily)
        data.index = data.index.droplevel(-1)
        self.data = data
        
        
class MonthlyCloud(CloudParameterizerBase):
    """Parameterizer to calculate the monthly cloud values from daily cloud"""
    
    name = 'monthly_cloud'
    
    summary = 'Calculate the monthly cloud values from daily cloud data'
    
    _datafile = 'monthly_cloud.csv'
    
    dbname = 'monthly_cloud'
    
    requires = ['daily_cloud']

    @staticmethod
    def calculate_monthly(df):
        return pd.DataFrame.from_dict(OrderedDict([
            ('wet_days', [df.wet_day.sum()]),
            ('mean_cloud_wet', [df.mean_cloud[df.wet_day].mean()]),
            ('mean_cloud_dry', [df.mean_cloud[~df.wet_day].mean()]),
            ('mean_cloud', [df.mean_cloud.mean()]),
            ('sd_cloud_wet', [df.mean_cloud[df.wet_day].std()]),
            ('sd_cloud_dry', [df.mean_cloud[~df.wet_day].std()]),
            ('sd_cloud', [df.mean_cloud.std()]),
            ]))
        
    def setup_from_file(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month']
        return super(MonthlyGHCNData, self).setup_from_file(*args, **kwargs)

    def setup_from_db(self, *args, **kwargs):
        kwargs['index_col'] = ['id', 'year', 'month']
        return super(MonthlyGHCNData, self).setup_from_db(*args, **kwargs)

    def setup_from_scratch(self):
        df = self.daily_cloud.data
        g = df.groupby(level=['id', 'year', 'month'])
        data = g.apply(self.calculate_monthly)
        data.index = data.index.droplevel(-1)
        # number of records per month
        df_nums = g.count().reset_index()
        df_nums['day'] = 1
        s = pd.to_datetime(df_nums[['year', 'month', 'day']])
        df_nums['ndays'] = ((s + pd.datetools.thisMonthEnd)[0] - s[0]).days + 1
        cols = ['wet_day', 'tmin', 'tmax', 'mean_cloud']
        complete_cols = [col + '_complete' for col in cols]
        for col, tcol in zip(cols, complete_cols):
            df_nums[tcol] = df_nums[col] == df_nums.ndays
        df_nums.drop('day', 1, inplace=True)
        self.data = data.merge(
            df_nums[complete_cols], left_index=True, right_index=True)
        
class CompleteDailyCloud(DailyCloud):
    """The parameterizer that calculates the days in complete months of cloud
    data"""

    name = 'cdaily_cloud'

    requires = ['daily_cloud', 'monthly_cloud']

    _datafile = "complete_daily_cloud.csv"

    dbname = 'complete_daily_cloud'

    summary = "Get the days of the complete daily cloud months"
    
    def init_from_scratch(self):
        pass

    def setup_from_scratch(self):
        monthly = self.monthly_cloud.data
        cols = ['wet_day', 'tmin', 'tmax', 'mean_cloud']
        complete_cols = [col + '_complete' for col in cols]
        self.data = self.daily_cloud.data.reset_index().merge(
            monthly[
                monthly[complete_cols].values.all(axis=1)][[]].reset_index(),
            how='inner', on=['id', 'year', 'month'], copy=False).set_index(
                ['id', 'year', 'month', 'day'])
        

def cloud_func(x, a):
    """Function for fitting the mean of wet and dry cloud to the mean of all 
    cloud
    
    This function returns y with
    
    .. math::
        y = ((-a - 1) / (a^2 x - a^2 - a)) - \\frac{1}{a}
        
    Parameters
    ----------
    x: np.ndarray
        The x input data
    a: float
        The parameter as mentioned in the equation above"""
    return ((-a - 1) / (a*a*x - a*a - a)) - 1/a


def cloud_sd_func(x, a):
    """Function for fitting the standard deviation of wet and dry cloud to the 
    mean of wet or dry cloud
    
    This function returns y with
    
    .. math::
        y = a^2 x (1 - x)
        
    Parameters
    ----------
    x: np.ndarray
        The x input data
    a: float
        The parameter as mentioned in the equation above"""
    return a * a * x * (1 - x)
    
    
class CompleteMonthlyCloud(MonthlyCloud):
     """Parameterizer to extract the months with complete clouds"""
     
     def setup_from_scratch(self):
        cols = ['wet_day', 'tmin', 'tmax', 'mean_cloud']
        complete_cols = [col + '_complete' for col in cols]
        self.data = self.monthly_cloud.data[
            self.monthly_cloud.data[complete_cols].values.all(axis=1)]
        
        
class CloudParameterizer(MonthlyCloud):
    """Parameterizer to extract the months with complete clouds"""
    
    name = 'cloud'
    
    summary = 'Parameterize the cloud data'
    
    requires =['monthly_cloud']

    _datafile = 'complete_monthly_cloud.csv'
    
    dbname = 'complete_monthly_cloud'
    
    fmt = dict(
        legend={'loc': 'upper left'},
        cmap='w_Reds',
        precision=0.1,
        xrange=(0, ['rounded', 95]),
        yrange=(0, ['rounded', 95]),
        legendlabels=['std. error of a: %(a_err)1.4f'],
        bounds=['minmax', 11, 0, 99],
        cbar='',
        bins=100,
        xlabel='on %(state)s days'
        )

    @property
    def ds(self):
        """The dataframe of this parameterization task converted to a dataset
        """
        import xarray as xr
        ds = xr.Dataset.from_dataframe(self.data)
        for state in ['wet', 'dry', 'all']:
            name = 'cloud' + (('_' + state) if state != 'all' else '')
            ds['mean_' + name].attrs['long_name'] = 'mean cloud fraction'
            ds['mean_' + name].attrs['state'] = state
            ds['sd_' + name].attrs['long_name'] = 'std. dev. of cloud fraction'
            ds['sd_' + name].attrs['state'] = state
        return ds
    

    @docstrings.dedent
    def run(self, plot_output=None, nc_output=None, project_output=None,
            new_project=False, project=None):
        """
        Run the parameterization
        
        Parameters
        ----------
        %(Parameterizer.run.parameters)s
        
        Returns
        -------
        %(Parameterizer.run.returns)s
        """
        import seaborn as sns
        import psyplot.project as psy
        import matplotlib.pyplot as plt
        self.logger.info('Calculating %s parameterization', self.name)
        
        sns.set_style("white")
        
        ds = self.ds
        
        ret_nml = {}
        ret_info = OrderedDict()
        ret_nml['weathergen'] = nml = {}
        
        # ---- file names
        nc_output = nc_output or self.nc_file
        plot_output = plot_output or self.pdf_file
        project_output = project_output or self.project_file
        ret_info['nc_file'] = nc_output
        ret_info['plot_file'] = plot_output
        ret_info['project_file'] = project_output

        sns.set_style("white")
        variables = ['cloud']
        states = ['wet' ,'dry']
        types = ['mean', 'sd']
        fit_funcs = {'mean': cloud_func, 'sd': cloud_sd_func}
        # ---- create project
        if not new_project and osp.exists(project or project_output):
            sp = psy.Project.load_project(project or project_output, 
                                          datasets=[ds])
        else:
            axes = np.concatenate([
                plt.subplots(1, 2, figsize=(12, 4))[1] for _ in range(2)])
            for fig in set(ax.get_figure() for ax in axes):
                fig.subplots_adjust(bottom=0.25)
            middle = (
                axes[0].get_position().x0 + axes[1].get_position().x1) / 2.
            axes = iter(axes)
            for v, t in product(variables, types):
                psy.plot.densityreg(
                    ds, name='%s_%s_wet' % (t, v), ax=next(axes), 
                    ylabel='%(long_name)s\non %(state)s days',
                    text=[(middle, 0.03, '%(long_name)s', 'fig', dict(
                         weight='bold', ha='center'))], fmt=self.fmt,
                    fit=fit_funcs[t])
                psy.plot.densityreg(
                    ds, name='%s_%s_dry' % (t, v), ax=next(axes),
                    ylabel='on %(state)s days', fmt=self.fmt,
                    fit=fit_funcs[t])
            sp = psy.gcp(True)[:]
        
        for v, t, state, key in product(
                variables, types, states, ['rsquared', 'slope', 'intercept']):
            vname = '%s%s_%s' % (v, t, state)
            nml_name = v + ('_sd' if t else '') + '_' + state[0]
            ret_info[vname] = vinfo = {}
            vinfo[key] = float(sp.plotters[0].plot_data[1].attrs[key])
            plotter = sp(name=vname).plotters[0]
            nml[nml_name + '1'] = float(
                plotter.plot_data[1].attrs.get('intercept', 0))
            nml[nml_name + '2'] = float(
                plotter.plot_data[1].attrs.get('slope'))
        
        
        # ---- save data and project
        sp.export(plot_output, tight=True)
        sp.save_project(project_output, 
                        use_rel_paths=True, paths=[nc_output])
        sp.close(True, True)  
        
        return ret_nml, ret_info
        
        
cr.pickle(Parameterizer, Parameterizer._get_copy_reg)
