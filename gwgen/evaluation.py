# -*- coding: utf-8 -*-
"""Evaluation module of the gwgen module"""
import os
import os.path as osp
import inspect
import abc
import six
from psyplot.compat.pycompat import OrderedDict
from itertools import chain, product, starmap
import numpy as np
from scipy import stats
import pandas as pd
from gwgen.utils import docstrings
import gwgen.utils as utils
import logging

try:
    import copyreg as cr
except ImportError:
    import copy_reg as cr
    

class EvaluatorMeta(abc.ABCMeta):
    """Meta class for the :class:`Parameterizer`"""

    def __new__(cls, name, bases, namespace):
        new_cls = super(EvaluatorMeta, cls).__new__(
            cls, name, bases, namespace)
        if new_cls.name:
            new_cls._registry[new_cls.name] = new_cls
        cr.pickle(new_cls, new_cls._get_copy_reg)
        return new_cls


@six.add_metaclass(EvaluatorMeta)
class Evaluator(object):
    """Abstract base class for evaluation tasks
    
    Evaluation tasks should incorporate a run method that is called by the
    :meth:`gwgen.main.ModelOrganizer.evaluate method"""
    
    fmt = {}

    _registry = {}

    name = None
    
    summary = ''
    
    _logger = None

    @property
    def logger(self):
        """The logger of this organizer"""
        return self._logger or logging.getLogger('.'.join(
            [__name__, self.__class__.__name__, self.name or '']))
        
    @property
    def data_dir(self):
        """str. Path to the directory where the source data of the model
        is located"""
        return self.model_config['data']
        
    @property
    def eval_dir(self):
        """str. Path to the directory were the processed data is stored"""
        
        ret = self.config.setdefault('evaldir', osp.join(
            self.config['expdir'], 'evaluation'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret
        
    @property
    def input_dir(self):
        """str. Path to the directory were the input data is stored"""
        ret = self.config.setdefault('indir', osp.join(
            self.config['expdir'], 'input'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret
    
    @property
    def reference_path(self):
        """The path to the reference file in the configuration"""
        return self.config.get(
            'reference', osp.join(self.eval_dir, 'reference.csv'))
        
    @reference_path.setter
    def reference_path(self, value):
        self.config['reference'] = value

    @property
    def df_ref(self):
        """The reference data frame"""
        df = pd.read_csv(self.reference_path, 
                         index_col=['id', 'year', 'month', 'day'])
        stations = list(self.stations)
        if len(stations) == 1:
            stations = slice(stations[0], stations[0])
        return df.loc(axis=0)[stations]

    @property
    def input_path(self):
        """The path to the model input file in the configuration"""
        return self.config.get(
            'input', osp.join(self.input_dir, 'input.csv'))
        
    @input_path.setter
    def input_path(self, value):
        self.config['input'] = value
        
    @property
    def output_dir(self):
        """str. Path to the directory were the input data is stored"""
        ret = self.config.setdefault('outdir', osp.join(
            self.config['expdir'], 'output'))
        if not osp.exists(ret):
            try:
                os.makedirs(ret)
            except FileExistsError:
                pass
        return ret
        
    @property
    def output_path(self):
        """The path to the model output file in the configuration"""
        return self.config['outdata']

    @property
    def nc_file(self):
        """NetCDF file for the project"""
        return osp.join(self.eval_dir, self.name + '.nc')
        
    @property
    def project_file(self):
        """Pickle file for the project"""
        return osp.join(self.eval_dir, self.name + '.pkl')
        
    @property
    def pdf_file(self):
        """pdf file with figures the project"""
        return osp.join(self.eval_dir, self.name + '.pdf')
    
    @property
    def df_sim(self):
        """The data frame of the simulation output"""
        df = pd.read_csv(self.output_path, 
                         index_col=['id', 'year', 'month', 'day'])
        stations = list(self.stations)
        if len(stations) == 1:
            stations = slice(stations[0], stations[0])
        return df.loc(axis=0)[stations]
        
    @docstrings.get_sectionsf('Evaluator')
    def __init__(self, stations, config, model_config, global_config={},
                 logger=None):
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
        """
        self.stations = stations
        self.config = config
        self.model_config = model_config
        self.global_config = global_config
        if isinstance(logger, six.string_types):
            logger = logging.getLogger(logger)
        self._logger = logger
        # overwrite the class attribute of the formatoptions
        self.fmt = self.fmt.copy()
        
    @staticmethod
    def _get_copy_reg(parameterizer):
        if parameterizer._engine is None:
            engine = None
        else:
            engine = parameterizer.engine.url 
        return parameterizer.__class__, (
            parameterizer.stations, parameterizer.config, 
            parameterizer.model_config, parameterizer.logger.name, engine)
        
    @abc.abstractmethod
    @docstrings.get_sectionsf('Parameterizer.run', 
                              sections=['Parameters', 'Returns'])
    @docstrings.dedent
    def run(self, plot_output=None, nc_output=None, project_output=None,
            new_project=False, project=None):
        """
        Run the evaluation task
        
        This method needs to be implemented by any evaluation task and can. It
        can take any keyword argument you want, but you should consider the
        :meth:`_modify_parser` method that can be used to modify the parser 
        according to the needs
        
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
        self.logger.info('Calculating %s evaluation', self.name)
        ret_info = OrderedDict()

        # ---- file names
        nc_output = nc_output or self.nc_file
        plot_output = plot_output or self.pdf_file
        project_output = project_output or self.project_file
        ret_info['nc_file'] = nc_output
        ret_info['plot_file'] = plot_output
        ret_info['project_file'] = project_output

        # ---- open dataset
        ds = self.ds
        
        # ---- create project
        if not new_project and osp.exists(project or project_output):
            import psyplot.project as psy
            self.logger.debug('    Loading existing project %s', 
                              project or project_output)
            sp = psy.Project.load_project(
                project or project_output, datasets=[ds])
        else:
            self.logger.debug('    Creating project...')
            sp = self.create_project(ds)
        
        # ---- save data and project
        pdf = sp.export(plot_output, tight=True, close_pdf=False)
        self.logger.debug('    Saving project to %s', project_output)
        sp.save_project(project_output, use_rel_paths=True, 
                        paths=[nc_output])
        
        # ---- make plots not covered by psyplot
        self.plot_additionals(pdf)

        # ---- configure the experiment
        self.make_run_config(sp, ret_info)

        # ---- export the figures
        self.logger.debug('    Saving plots to %s', plot_output)
        pdf.close()
        
        # ---- close the project
        sp.close(True, True)
        self.logger.debug('Done.')
        
        return ret_info

    @docstrings.get_sectionsf('Parameterizer.create_project')
    @docstrings.dedent
    def create_project(self, ds):
        """
        To be reimplemented for each parameterization task
        
        Parameters
        ----------
        ds: xarray.Dataset
            The dataset to plot"""
        import psyplot.project as psy
        return psy.gcp()
        
    @docstrings.get_sectionsf('Parameterizer.make_run_config')
    @docstrings.dedent
    def make_run_config(self, sp, info):
        """
        Method to be reimplemented to provide information about the evaluation
        task
        
        Parameters
        ----------
        sp: psyplot.project.Project
            The project of the data
        info: dict
            The dictionary for saving additional information of the 
            parameterization task"""
        return
        
    @docstrings.get_sectionsf('Parameterizer.plot_additionals')
    @docstrings.dedent
    def plot_additionals(self, pdf):
        """
        Method to be reimplemented to make additional plots (if necessary)
        
        Parameters
        ----------
        pdf: matplotlib.backends.backend_pdf.PdfPages
            The PdfPages instance which can be used to save the figure
        """
        return

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
        
    @classmethod
    def get_task(cls, name):
        return cls._registry[name]

        
class EvaluationPreparation(Evaluator):
    """Evaluation task to prepare the evaluation"""
    
    name = 'prepare'
    
    summary = 'Prepare the for experiment for evaluation'
    
    http_stations = (
        'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt')

    @property
    def station_list(self):
        fname = osp.join(self.data_dir, 'ghcn', 'ghcnd-stations.txt')
        if not osp.exists(fname):
            self.logger.info("Downloading station file from %s to %s",
                             self.http_stations, fname)
            utils.download_file(self.http_stations, fname)
        station_ids, lat, lon = np.loadtxt(
            fname, usecols=[0, 1, 2], dtype='S11', unpack=True).astype(np.str_)
        return pd.DataFrame({'id': station_ids, 'lat': lat.astype(float), 
                             'lon': lon.astype(float)}, 
                            index=np.arange(len(station_ids)))
    
    @docstrings.dedent
    def run(self, ref_path=None, input_path=None, store_raw=False, 
            setup_from=None):
        """
        Get the input data for the evaluation
        
        Parameters
        ----------
        ref_path: str
            The path of the file where to store the reference data. If None and
            not already set in the configuration, it will default to 
            ``'evaluation/reference.csv'``
        input_path: str
            The path of the file where to store the model input. If None, and
            not already set in the configuration, it will default to 
            ``'inputdir/input.csv'`` where *inputdir* is the path to the
            input directory (by default, *input* in the experiment directory)
        store_raw: bool
            If True, set, the raw input data is stored as well
        setup_from: { 'scratch' | 'file' }
            How to setup the raw data. By default (if the data has already)
            been saved to a file, it is set up from file. Otherwise from 
            scratch
            """
        def get(name):
            return next(t for t in tasks if t.name == name)
        from gwgen.parameterization import (
            Parameterizer, CompleteDailyCloud, CompleteDailyGHCNData, 
            CompleteMonthlyCloud, CompleteMonthlyGHCNData)
        classes = [CompleteDailyCloud, CompleteDailyGHCNData, 
                   CompleteMonthlyCloud, CompleteMonthlyGHCNData]
        config = self.config.copy()
        config['paramdir'] = self.eval_dir
        kws = dict(config=config, model_config=self.model_config,
                   to_csv=store_raw, setup_from=setup_from)
        base_kws = {cls.name: kws for cls in classes}
        tasks = Parameterizer.setup_tasks(
            self.stations, base_kws=base_kws, global_conf=self.global_config,
            logger=self.logger)
        cday = get(CompleteDailyGHCNData.name).data
        ccday = get(CompleteDailyCloud.name).data
        cmonth = get(CompleteMonthlyGHCNData.name).data
        ccmonth = get(CompleteMonthlyCloud.name).data
        reference = cday.merge(
            ccday[['mean_cloud']], left_index=True, right_index=True).reset_index()
        exp_input = cmonth.merge(
            ccmonth[['mean_cloud']], left_index=True, right_index=True)
        exp_input = exp_input.reset_index().merge(
            self.station_list, on='id')
        reference['wet_day'] = (reference.prcp > 0).astype(int)
        
        # save the data and store the paths in the configuration
        self.reference_path = ref_path = ref_path or self.reference_path
        self.input_path = input_path = input_path or self.input_path

        order = ['tmin', 'tmax', 'mean_cloud', 'prcp', 'wet_day']
        self.logger.debug('Exporting reference data to %s', ref_path)
        reference[['id', 'year', 'month', 'day'] + order].to_csv(
            ref_path, index=False)
        self.logger.debug('Exporting input data to %s', input_path)
        exp_input[['id', 'lon', 'lat', 'year', 'month'] + order].to_csv(
            input_path, index=False, float_format='%6.2f')
        
    @classmethod
    def _modify_parser(cls, parser):
        parser.update_arg('ref_path', short='r')
        parser.update_arg('setup_from', short='f', long='from')
        parser.update_arg('input_path', short='i')
        parser.update_arg('store_raw', short='store')
        
        
class QuantileEvaluation(Evaluator):
    """Evaluator to evaluate specific quantiles"""
    
    name = 'quants'
    
    summary = 'Compare the quantiles of simulation and observation'
    
    names = OrderedDict([
        ('prcp', {'long_name': 'Precipitation',
                  'units': 'mm'}), 
        ('tmin', {'long_name': 'Min. Temperature',
                  'units': 'degC'}),
        ('tmax', {'long_name': 'Max. Temperature',
                  'units': 'degC'}), 
        ('mean_cloud', {'long_name': 'Cloud fraction',
                        'units': '-'})
        ])

    all_variables = [[v + '_ref', v + '_sim'] for v in names]

    _quantiles = [25, 50, 75, 90, 95, 99, 100]

    _round = True
    
    #: default formatoptions for the 
    #: :class:`psyplot.plotter.linreg.DensityRegPlotter` plotter
    fmt = kwargs = dict(
        legend={'loc': 'upper left'},
        cmap='w_Reds',
        title='%(pctl)sth percentile',
        xlabel='%(type)s {desc}',
        ylabel='%(type)s {desc}',
        xrange=(['minmax', 1], ['minmax', 99]),
        yrange=(['minmax', 1], ['minmax', 99]),
        legendlabels=['$R^2$ = %(rsquared)s'],
        bounds=['minmax', 11, 0, 99],
        cbar='',
        bins=10,
        )
    
    @property
    def data(self):
        """The dataframe containing the observed and simulated data"""
    
        # create reference dataframe
        df_ref = self.df_ref
        # create simulation dataframe
        df_sim = self.df_sim
        names = self.names
        # load observed precision
        if self._round:
            for name in set(names) - {'mean_cloud'}:
                df_sim[name].values[:] = self.round_to_ref_prec(
                    df_ref[name].values, df_sim[name].values)
        # merge reference and simulation into one single dataframe
        df = df_ref.merge(df_sim, left_index=True, right_index=True,
                          suffixes=['_ref', '_sim'])
        if 'mean_cloud' in names:
            from gwgen.parameterization import CloudParameterizerBase
            # mask out non-complete months for cloud validation
            df_map = CloudParameterizerBase.eecra_ghcn_map()
            df_map['complete'] = True
            df.reset_index(['year', 'month', 'day'], inplace=True)
            df = df.merge(df_map, left_index=True, right_index=True)
            cloud_names = ['mean_cloud_ref', 'mean_cloud_sim']
            df.ix[df.complete.isnull(), cloud_names] = np.nan
            df.set_index(['year', 'month', 'day'], append=True, inplace=True)
        # calculate the percentiles for each station and month
        g = df.sort_index().groupby(level=['id', 'year'])
        ret = g.apply(self.calc)
        ret.index = ret.index.droplevel(2)
        return ret
    
    @property
    def ds(self):
        """The dataset of the quantiles"""
        import xarray as xr
        ds = xr.Dataset.from_dataframe(
            self.data.reset_index().set_index('pctl', append=True).swaplevel())
        for orig, attrs, (vref, vsim) in zip(
                self.names, self.names.values(), self.all_variables):
            ds[vsim].attrs.update(attrs)
            ds[vref].attrs.update(attrs)
            ds[vsim].attrs['standard_name'] = orig
            ds[vref].attrs['standard_name'] = orig
            ds[vref].attrs['type'] = 'observed'
            ds[vsim].attrs['type'] = 'simulated'
        ds.pctl.attrs['long_name'] = 'Percentile'
        return ds
    
    @docstrings.dedent
    def run(self, quantiles=[25, 50, 75, 90, 95, 99, 100], no_rounding=False,
            *args, **kwargs):
        """
        Run the quantile evaluation
        
        Parameters
        ----------
        quantiles: list of floats
            The quantiles to use for calculating the percentiles
        no_rounding: bool
            Do not round the simulation to the infered precision of the 
            reference. The infered precision is the minimum difference between
            two values with in the entire data
            """
        self._quantiles = quantiles
        self._round = not no_rounding
        return super(QuantileEvaluation, self).run(*args, **kwargs)
        
    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(super(QuantileEvaluation, cls).run)
        parser.update_arg('quantiles', short='q')
        super(QuantileEvaluation, cls)._modify_parser(parser)
        
    def create_project(self, ds):
        import psyplot.project as psy
        import seaborn as sns
        sns.set_style('white')
        for vref, vsim in self.all_variables:
            self.logger.debug('Creating plots of %s', vsim)
            kwargs = dict(precision=0.1) if vref.startswith('prcp') else {}
            psy.plot.densityreg(ds, name=vsim, coord=vref, fmt=self.fmt, 
                                pctl=range(ds.pctl.size), **kwargs)
        return psy.gcp(True)[:]
        
    def make_run_config(self, sp, info):
        for orig in self.names:
            info[orig] = d = OrderedDict()
            for plotter in sp(standard_name=orig).plotters:
                d[int(plotter.data.pctl.values)] = pctl_d = OrderedDict()
                for key in ['rsquared', 'slope', 'intercept']:
                        val = plotter.plot_data[1].attrs.get(key)
                        if val is not None:
                            pctl_d[key] = float(val)
        return info
                

    def calc(self, group):
        def calc_percentiles(vname):
            arr = group[vname].values
            arr = arr[~np.isnan(arr)]
            if vname.startswith('prcp'):  # or vname.startswith('cloud'):
                arr = arr[arr > 0]
            if len(arr) == 0:
                return np.array([np.nan] * len(self._quantiles))
            else:
                return np.percentile(arr, self._quantiles)
        df = pd.DataFrame.from_dict(dict(zip(
            chain(*self.all_variables), map(calc_percentiles, 
                                            chain(*self.all_variables)
            ))))
        df['pctl'] = self._quantiles
        df.set_index('pctl')
        return df
            
    @staticmethod
    def round_to_ref_prec(ref, sim, func=np.ceil):
        """Round one array to the precision of another
        
        Parameters
        ----------
        ref: np.ndarray
            The reference array to get the precision from
        sim: np.ndarray
            The simulated array to round
        func: function
            The rounding function to use
            
        Returns
        -------
        np.ndarray
            Rounded `sim`"""
        ref_sorted = np.unique(ref)
        precision = (ref_sorted[1:] - ref_sorted[:-1]).min()
        return func((sim / precision) * precision)
        
        
class KSEvaluation(QuantileEvaluation):
    """Evaluation using a Kolmogorov-Smirnoff test"""
    
    name = 'ks'
    
    summary = 'Perform a kolmogorov smirnoff test'
    
    @staticmethod
    def calc(group):
        def calc(v1, v2, name):
            if len(v1) <= 10 or len(v2) <= 10:
                return {
                    name + '_stat': [np.nan],
                    name + '_p': [np.nan],
                    name: [None],
                    'n' + name + '_sim': [np.nan],
                    'n' + name + '_ref': [np.nan]}
            statistic, p_value = stats.ks_2samp(v1, v2)
            n = np.sqrt((len(v1) + len(v2)) / (len(v1) * len(v2)))
            return {
                name + '_stat': [statistic],
                name + '_p': [p_value],
                name: [statistic > 1.36 * n],
                'n' + name + '_sim': [len(v1)],
                'n' + name + '_ref': [len(v2)]}
        prcp_sim = group.prcp_sim.values[group.prcp_sim.values > 0]
        prcp_ref = group.prcp_ref.values[group.prcp_ref.values > 0]
        tmin_sim = group.tmin_sim.values[
            (group.tmin_sim.values < 100) & (group.tmin_sim.values > -100) &
            (~np.isnan(group.tmin_sim.values))]
        tmin_ref = group.tmin_ref.values[
            (group.tmin_ref.values < 100) & (group.tmin_ref.values > -100) &
            (~np.isnan(group.tmin_ref.values))]
        tmax_sim = group.tmax_sim.values[
            (group.tmax_sim.values < 100) & (group.tmax_sim.values > -100) &
            (~np.isnan(group.tmax_sim.values))]
        tmax_ref = group.tmax_ref.values[
            (group.tmax_ref.values < 100) & (group.tmax_ref.values > -100) &
            (~np.isnan(group.tmax_ref.values))]
        cloud_sl = group.mean_cloud_sim.notnull().values
        cloud_sim = group.mean_cloud_sim.values[cloud_sl]
        cloud_ref = group.mean_cloud_ref.values[cloud_sl]
        return pd.DataFrame.from_dict(dict(chain(*map(six.iteritems, starmap(
            calc, [(prcp_sim, prcp_ref, 'prcp'),
                   (tmin_sim, tmin_ref, 'tmin'),
                   (tmax_sim, tmax_ref, 'tmax'),
                   (cloud_sim, cloud_ref, 'mean_cloud')])))))
        
    
    def run(self):
        """Run the evaluation
        
        Returns
        -------
        dict
            The percentage of stations with no significant difference"""
        def significance_fractions(vname):
            return 100. - (len(df[vname][df[vname].notnull() & (df[vname])]) /
                           df[vname].count())*100.
        logger = self.logger                       
        logger.info('Calculating %s evaluation', self.name)
        
        ret = OrderedDict()
        df = self.data
        names = ['prcp', 'tmin', 'tmax', 'mean_cloud']
        for name in names:
            ret[name] = float(significance_fractions(name))
        
        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug('Done. Stations without significant difference:')
            for name, val in ret.items():
                logger.debug('    %s: %6.3f %%' % (name, val))
        return ret
        
    @classmethod
    def _modify_parser(cls, parser):
        pass
    

class SimulationQuality(Evaluator):
    """Evaluator to provide one value characterizing the quality of the 
    experiment
    
    The applied metric follows the formula
    
    .. math::
        
        m = \prod_{q\in Q}[(1 - R^2_q)|1 - a_q|] \cdot ks
        
    where :math:`q\in Q` are the quantiles from the quantile evaluation, 
    :math:`R^2_q` the coefficient of determination and :math:`a_q` the slope
    of quantile :math:`q`. In other words, it is the multiplication of the
    coefficients of determination, the deviation from the ideal slope 
    (:math:`a_q == 1`) and a 100 % aggreement."""
    
    pass