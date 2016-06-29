import os
import os.path as osp
import six
import sys
import logging
import logging.config
import yaml
from psyplot.docstring import dedent
from psyplot.compat.pycompat import OrderedDict


def _get_home():
    """Find user's home directory if possible.
    Otherwise, returns None.

    :see:  http://mail.python.org/pipermail/python-list/2005-February/325395.html

    This function is copied from matplotlib version 1.4.3, Jan 2016
    """
    try:
        if six.PY2 and sys.platform == 'win32':
            path = os.path.expanduser(b"~").decode(sys.getfilesystemencoding())
        else:
            path = os.path.expanduser("~")
    except ImportError:
        # This happens on Google App Engine (pwd module is not present).
        pass
    else:
        if os.path.isdir(path):
            return path
    for evar in ('HOME', 'USERPROFILE', 'TMP'):
        path = os.environ.get(evar)
        if path is not None and os.path.isdir(path):
            return path
    return None


def get_configdir(name):
    """
    Return the string representing the configuration directory.

    The directory is chosen as follows:

    1. If the ``name.upper() + CONFIGDIR`` environment variable is supplied,
       choose that.

    2a. On Linux, choose `$HOME/.config`.

    2b. On other platforms, choose `$HOME/.matplotlib`.

    3. If the chosen directory exists, use that as the
       configuration directory.
    4. A directory: return None.

    Notes
    -----
    This function is taken from the matplotlib [1] module

    References
    ----------
    [1]: http://matplotlib.org/api/"""
    configdir = os.environ.get('%sCONFIGDIR' % name.upper())
    if configdir is not None:
        return os.path.abspath(configdir)

    p = None
    h = _get_home()
    if ((sys.platform.startswith('linux') or
         sys.platform.startswith('darwin')) and h is not None):
        p = os.path.join(h, '.config/' + name)
    elif h is not None:
        p = os.path.join(h, '.' + name)

    if not os.path.exists(p):
        os.makedirs(p)
    return p


@dedent
def setup_logging(default_path=None, default_level=logging.INFO,
                  env_key='LOG_GWGEN'):
    """
    Setup logging configuration

    Parameters
    ----------
    default_path: str
        Default path of the yaml logging configuration file. If None, it
        defaults to the 'logging.yaml' file in the config directory
    default_level: int
        Default: :data:`logging.INFO`. Default level if default_path does not
        exist
    env_key: str
        environment variable specifying a different logging file than
        `default_path` (Default: 'LOG_CFG')

    Returns
    -------
    path: str
        Path to the logging configuration file

    Notes
    -----
    Function taken from
    http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python"""
    path = default_path or os.path.join(
        os.path.dirname(__file__), 'logging.yaml')
    value = os.getenv(env_key, None)
    home = _get_home()
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        for handler in config.get('handlers', {}).values():
            if '~' in handler.get('filename', ''):
                handler['filename'] = handler['filename'].replace(
                    '~', home)
        logging.config.dictConfig(config)
    else:
        path = None
        logging.basicConfig(level=default_level)
    return path


def ordered_yaml_load(stream, Loader=None, object_pairs_hook=OrderedDict):
    """Loads the stream into an OrderedDict.
    Taken from

    http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-
    mappings-as-ordereddicts"""
    Loader = Loader or yaml.Loader

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data, stream=None, Dumper=None, **kwds):
    """Dumps the stream from an OrderedDict.
    Taken from

    http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-
    mappings-as-ordereddicts"""
    Dumper = Dumper or yaml.Dumper

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def _config_prop(name, fattr, doc=""):

    def getx(self):
        fname = getattr(self, fattr)
        if hasattr(self, "_" + name):
            pass
        elif osp.exists(fname):
            with open(fname) as f:
                setattr(self, "_" + name, ordered_yaml_load(f))
        else:
            setattr(self, "_" + name, OrderedDict())
        return getattr(self, "_" + name)

    def setx(self, value):
        if self._store:
            fname = getattr(self, fattr)
            if osp.exists(fname):
                os.rename(fname, fname + '~')
            elif not osp.exists(osp.dirname(fname)):
                os.makedirs(osp.dirname(fname))
            with open(fname, 'w') as f:
                ordered_yaml_dump(value, f, default_flow_style=False)
        else:
            setattr(self, "_" + name, value)

    def delx(self):
        delattr(self, "_" + name)

    return property(getx, setx, delx, doc)


class Config(object):
    """Configuration class for one model organiser"""

    #: Boolean that is True when the experiments shall be synched with the
    #: files on the harddisk. Use the :meth:`save` method to store the
    #: configuration
    _sync = False

    def __init__(self, name):
        self.name = name
        self.conf_dir = get_configdir(name)
        self._exp_file = osp.join(self.conf_dir, 'experiments.yml')
        self._model_file = osp.join(self.conf_dir, 'models.yml')
        self._globals_file = osp.join(self.conf_dir, 'globals.yml')

    experiments = _config_prop(
        'experiments', '_exp_file', "The full meta data for all the runs")

    models = _config_prop(
        'models', '_model_file',
        "A mapping from the model names to their directories")

    global_config = _config_prop(
        'global_config', '_globals_file',
        "The global settings for the experiments")

    def save(self):
        self._store = True
        try:
            for attr in ['experiments', 'models', 'global_config']:
                setattr(self, attr, getattr(self, attr))
        except:
            raise
        finally:
            self._sync = False

setup_logging()
