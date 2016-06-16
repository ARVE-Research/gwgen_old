# -*- coding: utf-8 -*-
from _parseghcnrow import parseghcnrow
import six
from functools import partial
import logging
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from itertools import chain, starmap
import calendar
from scipy import stats
import argparse
import multiprocessing as mp
import datetime as dt
import warnings
import re

if six.PY2:
    from itertools import imap as map, izip as zip
    range = xrange

daymon_patt = re.compile(r'(?:\w|-){11}(\d{6})(?:TMAX|TMIN|PRCP)')

formatter = logging.Formatter(
    '%(levelname)s: - %(name)s - %(asctime)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger = logging.getLogger('multiprocessing')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def read_df(ifile):
    # get number of days in the file
    with open(ifile) as f:
        ndays = np.sum(list(map(ndaymon, np.unique(list(map(
            lambda m: m.group(1), filter(
                None, map(daymon_patt.match, f.xreadlines()))))))))
    stationid, dates, variables, flags, j = parseghcnrow.parse_station(
        ifile, ndays or 100)
    dates = dates[:j]
    variables = variables[:j].astype(np.float64)
    flags = flags[:j]
    flags = np.core.defchararray.replace(flags, ' ', '')
    variables[variables == -9999] = np.nan
    vlst = ['tmin', 'tmax', 'prcp']
    df = pd.DataFrame.from_dict(dict(chain(
        [('id', np.repeat([stationid], j))],
        zip(('year', 'month', 'day'), dates.T),
        zip(vlst, variables.T),
        chain(*[zip((var + '_m', var + '_q', var + '_s'), arr)
                for var, arr in zip(vlst, np.rollaxis(flags, 2, 1).T)]))))
    return df


def ndaymon(yearmon):
    year = int(yearmon[:4])
    month = int(yearmon[4:])
    d = dt.date(year, month, 1)
    d2 = d.replace(
        year=year + 1 if month == 12 else year,
        month=1 if month == 12 else month + 1)
    return (d2 - d).days


def summary(df):
    n = calendar.monthrange(df.year.values[0], df.month.values[0])[1]
    return pd.DataFrame.from_dict(
        {'tmin': [df.tmin.mean()], 'tmax': [df.tmax.mean()],
         'trange': [(df.tmax - df.tmin).mean()],
         'prcp': [df.prcp.sum()], 'tmin_abs': [df.tmin.min()],
         'tmax_abs': [df.tmax.max()], 'prcpmax': [df.prcp.max()],
         'tmin_complete': [df.tmin.count() == n],
         'tmax_complete': [df.tmax.count() == n],
         'prcp_complete': [df.prcp.count() == n]})


def prcp_dist_params(df, threshs=np.array([5, 7.5, 10, 12.5, 15, 17.5, 20])):
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


def _read_stations_from_file(fname, skip_header):
    return np.loadtxt(fname, usecols=[0], dtype=str)


def _read_table(table_name, engine):
    arr = pd.read_sql_table(table_name, engine, columns=['id']).values
    return arr.reshape((len(arr), ))


def worker(datapath, engine_str, return_daily, num_stations):
    def to_sql(dfs, table_name, insert_date=False):
        engine = create_engine(engine_str)
        df = pd.concat(dfs, ignore_index=True)
        if insert_date:
            df['date'] = np.array(
                ['{0[0]}-{0[1]:02d}-{0[2]:02d}'.format(x)
                 for x in zip(df.year, df.month, df.day)], dtype='datetime64')
            df = df[[col for col in df.columns
                     if col not in ['year', 'month', 'day']]]
        df.to_sql(table_name, engine, if_exists='append', index=False)

    def make_summary(df):
        out = df.groupby(['id', 'year', 'month']).apply(
            summary)
        out.index = out.index.droplevel(-1)
        return out.reset_index()

    def calc_prcp_params(df, df_sum):
        df = df[['id', 'year', 'month', 'prcp']].merge(
            df_sum[['id', 'year', 'month', 'prcp_complete']],
            on=['id', 'year', 'month'])
        return df[df.prcp_complete].groupby(['id', 'month']).apply(
            prcp_dist_params).reset_index()

    procNum, stations = num_stations
    logger = mp.get_logger().getChild('proc' + str(procNum))
    try:
        # -------------------------------------------------------------------------
        # ------------------------ read raw data ----------------------------------
        # -------------------------------------------------------------------------
        t = t0 = dt.datetime.now()
        logger.info('Processing %i stations', len(stations))
        dfs_root = list(map(
            read_df, map(datapath.format, stations)))
        logger.info('Done. Time needed: %s... Calculating summaries',
                    dt.datetime.now() - t)

        # -------------------------------------------------------------------------
        # -------------------- calculate monthly summaries ------------------------
        # -------------------------------------------------------------------------

        t = dt.datetime.now()
        summaries = list(map(make_summary, dfs_root))
        logger.info('Done. Time needed: %s. Calculating distribution '
                    'parameters', dt.datetime.now() - t)

        # -------------------------------------------------------------------------
        # ----------------- calculate distribution parameters ---------------------
        # -------------------------------------------------------------------------

        t = dt.datetime.now()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', 'Mean of empty slice', RuntimeWarning)
            dists = list(starmap(calc_prcp_params, zip(dfs_root, summaries)))
        logger.info('Done. Time needed: %s', dt.datetime.now() - t)

        # -------------------------------------------------------------------------
        # ----------------- write into database -----------------------------------
        # -------------------------------------------------------------------------
    #    lock.acquire()
    #    try:
    #        t = dt.datetime.now()
    #        logger.info('Writing into database')
    #        threads = [Thread(target=to_sql, args=args)
    #                   for args in [
    #                       # (dfs_root, 'calib_daily2', True),
    #                       # (summaries, 'calib_monthly2'),
    #                       (dists, 'prcp_dist_params2')]]
    #        for thread in threads:
    #            thread.start()
    #        for thread in threads:
    #            thread.join()
    #    except:
    #        raise
    #    finally:
    #        lock.release()
    #    logger.info('Done. Time needed: %s', dt.datetime.now() - t)
        logger.info('Total time needed: %s', dt.datetime.now() - t0)
        if return_daily:
            return pd.concat(summaries), pd.concat(dists), pd.concat(dfs_root)
        else:
            return pd.concat(summaries), pd.concat(dists), None
    except:
        logger.error('Error occured!', exc_info=True)
        raise


def init(l):
    global lock
    lock = l


def main(args=None):
    parser = argparse.ArgumentParser(
        description="""
        Parse stations from the GHCN data into columns of daily values and
        calculate the precipitation statistics based upon a hybrid
        gamma-gp distribution""")
    parser.add_argument(
        'ifile', metavar='file_or_tablename_or_stations', help="""
        Path to the input file containing 1 column corresponding
        to the station ids. Otherwise it can be a table name to be red from the
        `db` database (see the `from_db` option) or a list `station_id` to
        specify them manually (see the `s` option)""", nargs='+')
    parser.add_argument(
        '-s', '--stations', action='store_true', help="""
        Interprete the positional arguments as `station ids`.""")
    parser.add_argument(
        '-from_db', '--from_database', action='store_true', help="""
        Interprete the positional arguments as a table name.""")
    parser.add_argument(
        '-db', '--database', default='ghcn-daily_2016', help="""
        The database database to inject the data into. Default: %(default)s""")
    parser.add_argument(
        '-u', '--user', default='arve', help="""
        The databse user name. Default: %(default)s""")
    parser.add_argument(
        '-H', '--host', default='10.0.1.8', help="""
        The host of the database database. Default: %(default)s""")
    parser.add_argument(
        '-p', '--port', default=5432, type=int, help="""
        The port of the database database. Default: %(default)s""")
    parser.add_argument(
        '-e', '--engine', default='postgresql', help="""
        The engine to use when accessing the database. Default: %(default)s""")
    parser.add_argument(
        '-header', '--skip_header', type=int, default=0,
        help='Number of lines to skip in `ifile`. Default: %(default)i')
    parser.add_argument(
        '-dp', '--datapath',
        default='/Volumes/arve/shared/datasets/weather_station/ghcndaily_2016/rawdata/ghcnd_all/{0}.dly',
        help='Path template for the ghcn data. Default: %(default)s')
    parser.add_argument(
        '-tsum', '--target_db_mon', default='calib_monthly',
        help="Name of the data table for the monthly summaries")
    parser.add_argument(
        '-tdist', '--target_db_dist', default='prcp_dist_params',
        help=("Name of the data table for the precipitation distribution "
              "parameters"))
    parser.add_argument(
        '-tday', '--target_db_day', default=None,
        help="Name of the data table for the daily data")
    t = t0 = dt.datetime.now()
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    engine_str = '%(engine)s://%(user)s@%(host)s:%(port)s/%(database)s' % vars(
        args)
    if args.from_database:
        stations = _read_table(args.ifile[0], create_engine(engine_str))
    elif args.stations:
        stations = np.array(args.ifile, dtype=str)
    else:
        stations = _read_stations_from_file(args.ifile[0], args.skip_header)
    nprocs = None
    N = len(stations)
    logger.info("Processing %i stations in total...", N)
    pool = mp.Pool(nprocs, initializer=init, initargs=(mp.RLock(),))
    stations_per_proc = 500
    if len(stations) > stations_per_proc:
        stations = np.split(stations, np.arange(
            stations_per_proc, N, stations_per_proc, dtype=int))
    else:
        stations = [stations]
    func = partial(worker, args.datapath, engine_str,
                   args.target_db_day is not None)
    res = pool.map_async(func, enumerate(stations))
    engine = create_engine(engine_str)
    ret = res.get()
    res_dfs = [pd.concat([r[i] for r in ret]) for i, df in enumerate(ret[0])]
#    worker(args.datapath, vars(args), stations[0])
    logger.info('Total time needed for processing %i stations: %s', N,
                dt.datetime.now() - t)
    for attr, res_df in zip(
            ['target_db_mon', 'target_db_dist', 'target_db_day'], res_dfs):
        name = getattr(args, attr)
        if name:
            t = dt.datetime.now()
            logger.info('Writing into database %s', name)
            res_df.to_sql(name, engine, if_exists='append', index=False)
            logger.info('Done. Time needed: %s', dt.datetime.now() - t)
    logger.info('Total time needed for processing and writing %i stations: %s',
                N, dt.datetime.now() - t0)


if __name__ == '__main__':
    main()
