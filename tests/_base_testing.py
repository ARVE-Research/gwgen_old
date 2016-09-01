import os
import os.path as osp
import six
import shutil
import unittest
import glob
import gwgen
import numpy as np
import tempfile
from gwgen.main import ModelOrganizer
import gwgen.utils as utils
from gwgen.config import setup_logging

test_root = osp.abspath(osp.dirname(__file__))

_test_stations = osp.join(test_root, 'test_stations.dat')

_eecra_test_stations = osp.join(test_root, 'eecra_test_stations.dat')


setup_logging(osp.join(test_root, 'logging.yaml'))


dbname = 'travis_ci_test'


class BaseTest(unittest.TestCase):
    """Test the :class:`gwgen.main.ModuleOrganizer` class"""

    test_dir = None

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='tmp_gwgentest')
        os.environ['GWGENCONFIGDIR'] = self.config_dir = osp.join(
            self.test_dir, 'config')
        if not osp.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not osp.exists(self.config_dir):
            os.makedirs(self.config_dir)
        self.test_db = osp.basename(self.test_dir)
        self.stations_file = osp.join(self.test_dir,
                                      osp.basename(_test_stations))
        shutil.copyfile(_test_stations, self.stations_file)
        self.eecra_stations_file = osp.join(self.test_dir,
                                            osp.basename(_eecra_test_stations))
        shutil.copyfile(_eecra_test_stations, self.eecra_stations_file)
        self.organizer = ModelOrganizer('gwgen')
        global_conf = self.organizer.config.global_config
        global_conf['data'] = osp.dirname(__file__)
        global_conf['use_relative_links'] = False
        if use_db:
            self._clear_db()
            global_conf['database'] = dbname

    def tearDown(self):
        if osp.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if osp.exists(self.config_dir):
            shutil.rmtree(self.config_dir)
        if use_db:
            self._clear_db()
        del self.organizer
        del self.test_dir
        del self.config_dir

    def _clear_db(self):
        engine = utils.get_postgres_engine(dbname)[0]
        conn = engine.connect()
        for table in engine.table_names():
            conn.execute("DROP TABLE %s;" % table)
        conn.close()

    @property
    def stations(self):
        """A numpy array of the stations in :attr:`test_stations`"""
        return np.loadtxt(
            self.stations_file, dtype='S11', usecols=[0]).astype(
                np.str_)

    def _test_setup(self):
        """Test the setup of a model. We make this method private such that
        it is not called everytime"""
        self.organizer.setup(self.test_dir, 'test_model0', link=False)
        mpath = osp.join(self.test_dir, 'test_model0')
        self.assertTrue(osp.isdir(mpath))
        original_files = sorted(map(osp.basename, glob.glob(osp.join(
            osp.dirname(gwgen.__file__), 'src', '*.f90'))))
        copied_files = sorted(map(osp.basename, glob.glob(osp.join(
            mpath, 'src', '*.f90'))))
        self.assertEqual(original_files, copied_files)
        self.assertIn('test_model0', self.organizer.config.models)

        # createa new model and let it automatically assign the name
        self.organizer.setup(self.test_dir)
        mpath = osp.join(self.test_dir, 'test_model1')
        self.assertTrue(osp.isdir(mpath))
        original_files = sorted(map(osp.basename, glob.glob(osp.join(
            osp.dirname(gwgen.__file__), 'src', '*.f90'))))
        copied_files = sorted(map(osp.basename, glob.glob(osp.join(
            mpath, 'src', '*.f90'))))
        self.assertEqual(original_files, copied_files)
        self.assertIn('test_model1', self.organizer.config.models)

    def _test_init(self):
        """Test the intialization of a new experiment. We make this method
        private such that it is not called everytime"""
        self.organizer.setup(self.test_dir)
        modelname = self.organizer.modelname
        self.organizer.init(experiment='testexp0')
        expdir = osp.join(self.test_dir, modelname, 'experiments', 'testexp0')
        self.assertTrue(osp.exists(expdir),
                        msg='Experiment directory %s does not exist!' % expdir)
        self.assertIn('testexp0', self.organizer.config.experiments)

        # test without argument
        self.organizer.setup(self.test_dir)
        modelname = self.organizer.modelname
        self.organizer.init(experiment=None)
        expdir = osp.join(self.test_dir, modelname, 'experiments', 'testexp1')
        self.assertTrue(osp.exists(expdir),
                        msg='Experiment directory %s does not exist!' % expdir)
        self.assertIn('testexp1', self.organizer.config.experiments)

    @staticmethod
    def _test_url(url, *args, **kwargs):
        if six.PY3:
            from urllib import request
            request.urlopen(url, *args, **kwargs)
        else:
            import urllib
            urllib.urlopen(url, *args, **kwargs)

# check if we are online by trying to connect to google
try:
    BaseTest._test_url('https://www.google.de')
    online = True
except:
    online = False


# try to connect to a postgres database
try:
    utils.get_postgres_engine(dbname, create=True, test=True)
    use_db = True
except:
    use_db = False
