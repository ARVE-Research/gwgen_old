import six
import logging
from psyplot.docstring import DocStringProcessor

if six.PY2:
    from itertools import ifilterfalse as filterfalse


docstrings = DocStringProcessor()


def download_file(url, target=None):
    """Download a file from the internet

    Parameters
    ----------
    url: str
        The url of the file
    target: str or None
        The path where the downloaded file shall be saved. If None, it will be
        saved to a temporary directory

    Returns
    -------
    file_name: str
        the downloaded filename"""
    if six.PY3:
        from urllib import request
        return request.urlretrieve(url, target)[0]
    else:
        import urllib
        return urllib.urlretrieve(url, target)[0]


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    Function taken from https://docs.python.org/2/library/itertools.html"""
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


@docstrings.get_sectionsf('get_postgres_engine')
@docstrings.dedent
def get_postgres_engine(database, user=None, host='127.0.0.1', port=None):
    """
    Get the engine to access the given `database`

    This method creates an engine using sqlalchemy's create_engine function
    to access the given `database` via postgresql. If the database is not
    existent, it will be created

    Parameters
    ----------
    database: str
        The name of a psql database. If provided, the processed data will
        be stored
    user: str
        The username to use when logging into the database
    host: str
        the host which runs the database server
    port: int
        The port to use to log into the the database

    Returns
    -------
    sqlalchemy.engine.base.Engine
        Tha engine to access the database"""
    import sqlalchemy
    logger = logging.getLogger(__name__)
    base_str = 'postgresql://'
    if user:
        base_str += user + '@'
    base_str += host
    if port:
        base_str += ':' + port
    engine_str = base_str + '/' + database  # to create the database
    logger.debug("Creating engine with %s", engine_str)
    engine = sqlalchemy.create_engine(engine_str)
    logger.debug("Try to connect...")
    try:
        conn = engine.connect()
    except sqlalchemy.exc.OperationalError:
        # data base does not exist, so create it
        logger.debug("Failed...", exc_info=True)
        logger.debug("Creating database by logging into postgres")
        pengine = sqlalchemy.create_engine(base_str + '/postgres')
        conn = pengine.connect()
        conn.execute('commit')
        conn.execute('CREATE DATABASE ' + database)
        conn.close()
    else:
        conn.close()
    logger.debug('Done.')
    return engine, engine_str
