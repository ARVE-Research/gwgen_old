# -*- codng: utf-8 -*-
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
    