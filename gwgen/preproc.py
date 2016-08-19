# -*- coding: utf-8 -*-
"""Additional routines for preprocessing"""
import tempfile
import os.path as osp
from collections import namedtuple
import numpy as np
import pandas as pd
from psyplot.compat.pycompat import OrderedDict
import gwgen.utils as utils
from gwgen.utils import docstrings


class CloudGHCNMap(utils.TaskBase):
    """A task for computing the EECRA inventory for each station"""

    _registry = []

    name = 'eecra_ghcn_map'

    summary = 'Compute the inventory of the EECRA stations'

    http_xstall = 'http://cdiac.ornl.gov/ftp/ndp026c/XSTALL'

    _datafile = ['eecra_inventory.csv', 'eecra_ghcn_map.csv']

    dbname = ['eecra_inventory', 'eecra_ghcn_map']

    has_run = True

    @property
    def task_data_dir(self):
        return osp.join(self.data_dir, 'eecra')

    @property
    def default_config(self):
        return default_cloud_inventory_config()._replace(
            **super(CloudGHCNMap, self).default_config._asdict())

    @property
    def xstall_df(self):
        """The dataframe corresponding to the XSTALL stations"""
        use_xstall = self.task_config.xstall
        if utils.isstring(use_xstall):
            fname = self.task_config.no_xstall
        else:
            fname = tempfile.NamedTemporaryFile().name
            utils.download_file(self.http_xstall, fname)
        arr = np.loadtxt(fname, usecols=[1, 2, 3])
        df = pd.DataFrame(arr, columns=['station_id', 'lat', 'lon'])
        df['station_id'] = df.station_id.astype(int)
        df.set_index('station_id', inplace=True)
        return df

    @classmethod
    def _modify_parser(cls, parser):
        parser.setup_args(default_cloud_inventory_config)
        cls.has_run = False
        parser, setup_grp, run_grp = super(CloudGHCNMap, cls)._modify_parser(
            parser)
        parser.update_arg('xstall', group=setup_grp)
        parser.update_arg('max_distance', group=setup_grp, short='md')
        cls.has_run = True
        return parser, setup_grp, run_grp

    def __init__(self, *args, **kwargs):
        super(CloudGHCNMap, self).__init__(*args, **kwargs)
        self.__setup = False

    def setup(self, *args, **kwargs):
        self.__setup = True
        super(CloudGHCNMap, self).setup(*args, **kwargs)

    def init_from_scratch(self):
        from gwgen.parameterization import HourlyCloud
        task = HourlyCloud.from_task(self)
        task.download_src(task.raw_dir)  # make sure the source files exist

    def setup_from_scratch(self):
        from gwgen.parse_eecra import parse_file

        def compute(fname):
            return parse_file(fname).groupby('station_id')[
                ['lat', 'lon', 'year']].mean()

        self._set_data(pd.concat(list(map(compute, self.stations[:1]))), 0)
        self._set_data(pd.DataFrame([], columns=['station_id', 'distance'],
                                    index=pd.Index([], name='id')), 1)

    def write2db(self, *args, **kwargs):
        if self.__setup:
            return
        super(CloudGHCNMap, self).write2db(*args, **kwargs)

    def write2file(self, *args, **kwargs):
        if self.__setup:
            return
        super(CloudGHCNMap, self).write2file(*args, **kwargs)

    def run(self, info):
        from gwgen.evaluation import EvaluationPreparation
        from scipy.spatial import cKDTree
        import cartopy.crs as ccrs
        self.__setup = False
        eecra = self.data[0].groupby(level='station_id').agg(OrderedDict([
                ('lat', 'mean'), ('lon', 'mean'), ('year', ('min', 'max'))]))
        eecra.columns = ['lat', 'lon', 'firstyear', 'lastyear']

        use_xstall = self.task_config.xstall

        if use_xstall:
            to_replace = self.xstall_df
            eecra.loc[to_replace.index, ['lat', 'lon']] = to_replace

        t = EvaluationPreparation.from_task(self)
        # download inventory
        t.download_src()
        ghcn = t.station_list
        ghcn = ghcn.ix[ghcn.vname == 'PRCP']

        # transform to a coordinate system with metres as units
        eecra_points = ccrs.Mollweide().transform_points(
            ccrs.PlateCarree(), eecra.lon.values, eecra.lat.values)[..., :2]
        ghcn_points = ccrs.Mollweide().transform_points(
            ccrs.PlateCarree(), ghcn.lon.values, ghcn.lat.values)[..., :2]

        tree = cKDTree(eecra_points)
        distances, indices = tree.query(
            ghcn_points, distance_upper_bound=self.task_config.max_distance)
        out_of_range = np.isinf(distances)
        distances[out_of_range] = np.nan
        ghcn['station_id'] = eecra.iloc[indices - 1].index.values.astype(int)
        ghcn['distance'] = distances
        ghcn.ix[out_of_range, 'station_id'] = np.nan

        # remove duplicates in the station map
        ghcn.sort_values(['station_id', 'distance'], inplace=True)
        ghcn.ix[ghcn.duplicated('station_id'),
                ['station_id', 'distance']] = np.nan

        # and unnecessary columns
        ghcn.drop(['lat', 'lon', 'firstyr', 'lastyr', 'vname'], 1,
                  inplace=True)

        self.data = [eecra, ghcn]

        if not list(ghcn.index.names) == ['id']:
            ghcn.reset_index().set_index('id', inplace=True)

        if self.task_config.to_csv:
            self.write2file()
        if self.task_config.to_db:
            self.write2db()


CloudGHCNMapConfig = namedtuple(
    'CloudGHCNMapConfig',
    ['xstall', 'max_distance'] + list(utils.TaskConfig._fields))


CloudGHCNMapConfig = utils.append_doc(
    CloudGHCNMapConfig, docstrings.get_sections(docstrings.dedents("""
    Parameters
    ----------
    xstall: bool or str
        If True (default), download the XSTALL file from %s.
        This file contains some estimates of station longitude and latitude.
        If ``False`` or empty string, the file is not used, otherwise, if set
        with a string, it is interpreted as the path to the local file
    max_distance: float
        The maximum distance in meters for which we consider two stations as
        equal (Default: 1000m)
    %%(TaskConfig.parameters)s
    """ % CloudGHCNMap.http_xstall), 'CloudGHCNMapConfig'))


@docstrings.dedent
def default_cloud_inventory_config(xstall=True, max_distance=1000., *args,
                                   **kwargs):
    """
    Default config for :class:`CloudGHCNMap`

    Parameters
    ----------
    %(CloudGHCNMapConfig.parameters)s"""
    return CloudGHCNMapConfig(
        xstall, max_distance, *utils.default_config(*args, **kwargs))


# Alternative approach using dask. Currently much slower because we miss the
# agg function for dasks groupby
# class CloudInventory(utils.TaskBase):
#    """A task for computing the EECRA inventory for each station"""
#
#    _registry = []
#
#    name = 'inventory'
#
#    summary = 'Compute the inventory of the EECRA stations'
#
#    http_xstall = 'http://cdiac.ornl.gov/ftp/ndp026c/XSTALL'
#
#    _datafile = 'eecra_inventory.csv'
#
#    dbname = 'eecra_inventory'
#
#    has_run = False
#
#    setup_parallel = False  # is done in parallel via dask
#
#    @property
#    def task_data_dir(self):
#        return osp.join(self.data_dir, 'eecra')
#
#    @property
#    def default_config(self):
#        return default_cloud_inventory_config()._replace(
#            **super(CloudInventory, self).default_config._asdict())
#
#    @classmethod
#    def _modify_parser(cls, parser):
#        parser.setup_args(default_cloud_inventory_config)
#        cls.has_run = False
#        parser, setup_grp, run_grp = super(
#            CloudInventory, cls)._modify_parser(parser)
#        parser.update_arg('xstall', group=setup_grp)
#        cls.has_run = True
#        return parser, setup_grp, run_grp
#
#    def __init__(self, *args, **kwargs):
#        super(CloudInventory, self).__init__(*args, **kwargs)
#        self.__setup = False
#
#    def init_from_scratch(self):
#        from gwgen.parameterization import HourlyCloud
#        task = HourlyCloud.from_task(self)
#        task.download_src(task.raw_dir)  # make sure the source files exist
#
#    def setup_from_scratch(self):
#        from gwgen.parse_eecra import parse_file
#        import dask.dataframe as dd
#        from dask.multiprocessing import get
#        import pandas as pd
#
#        files = self.stations
#        df = parse_file(files[0])
#
#        divisions = [None] * (len(files) + 1)
#        dsk = {('eecra', i): (parse_file, fname)
#               for i, fname in enumerate(files)}
#        ddf = dd.DataFrame(dsk, 'eecra', df.columns, divisions)
#        g = ddf[['station_id', 'lat', 'lon', 'year']].groupby('station_id')
#
#        self.logger.debug('processing stations')
#        data = g[['lat', 'lon']].mean().compute(get=get)
#        self.logger.debug('Calculate first year')
#        data['firstyear'] = g.year.min().compute(get=get)
#        self.logger.debug('Calculate last year')
#        data['lastyear'] = g.year.max().compute(get=get)
#
#        use_xstall = self.task_config.xstall
#
#        if use_xstall:
#            if utils.isstring(use_xstall):
#                fname = self.task_config.no_xstall
#            else:
#                fname = tempfile.NamedTemporaryFile().name
#                utils.download_file(self.http_xstall, fname)
#            arr = np.loadtxt(fname, usecols=[1, 2, 3])
#            df = pd.DataFrame(arr, columns=['station_id', 'lat', 'lon'])
#            df['station_id'] = df.station_id.astype(int)
#            df.set_index('station_id', inplace=True)
#
#        self.data = data
