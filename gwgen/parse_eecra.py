# -*- codng: utf-8 -*-
import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict
from gwgen.utils import file_len
from gwgen._parseeecra import parseeecra

names = [
    'year', 'month', 'day', 'hour',
    'IB',
    'lat',
    'lon',
    'station_id',
    'LO',
    'ww',
    'N',
    'Nh',
    'h',
    'CL',
    'CM',
    'CH',
    'AM',
    'AH',
    'UM',
    'UH',
    'IC',
    'SA',
    'RI',
    'SLP',
    'WS',
    'WD',
    'AT',
    'DD',
    'EL',
    'IW',
    'IP']

def parse_file(ifile, year):
    """Parse a raw data file from EECRA and as a pandas DataFrame
    
    Parameters
    ----------
    ifile: str
        The raw (uncompressed) data file
    year: int
        The first year in the data file
        
    Returns
    -------
    pd.DataFrame
        `ifile` parsed into a dataframe"""
    df = pd.DataFrame.from_dict(OrderedDict(
        zip(names, parseeecra.parse_file(ifile, year, file_len(ifile)))))
    return df
    
def exctract_data(ids, src_dir, target_dir):
    """Extract the data for the given EECRA stations
    
    This function extracts the data for the given `ids` from the EECRA data
    base stored in  `src_dir` into one file for each *id* in `ids`. The 
    resulting filename will be like *id.csv*.
    
    Parameters
    ----------
    ids: np.ndarray of dtype int
        The numpy integer array
    src_dir: str
        The path to the source directory containing the raw (uncompressed) 
        EECRA database
    target_dir: str
        The path to the output directory
        
    Returns
    -------
    np.ndarray
        The paths of the filenames corresponding to ids"""
    ids = np.asarray(ids).astype(int)
    if ids.ndim == 0:
        ids.reshape((1,))
    parseeecra.extract_data(
        ids, osp.join(src_dir, ''), osp.join(target_dir, ''))
    return np.array([osp.join(src_dir, str(station_id) + '.csv')
                     for station_id in ids])