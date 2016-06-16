import os
import os.path as osp
import shutil
import unittest
from gwgen.main import ModelOrganizer
import gwgen
import glob


os.environ['GWGENCONFIGDIR'] = config_dir = 'config'


class OrganizerTest(unittest.TestCase):
    """Test the :class:`gwgen.main.ModuleOrganizer` class"""

    test_dir = 'test_dir'

    def setUp(self):
        if not osp.exists(self.test_dir):
            os.mkdir(self.test_dir)
        if not osp.exists(config_dir):
            os.mkdir(config_dir)
        self.organizer = ModelOrganizer('gwgen')

    def tearDown(self):
        if osp.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if osp.exists(config_dir):
            shutil.rmtree(config_dir)
        del self.organizer

    def test_setup(self):
        """Test the setup of a model"""
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

    def test_compile(self):
        """Test the compilation of a model"""
        self.organizer.setup(self.test_dir)
        modelname = self.organizer.modelname
        self.organizer.compile()
        self.assertTrue(osp.exists(osp.join(
            self.test_dir, modelname, 'bin', 'gwgen')))

    def test_init(self):
        """Test the intialization of a new experiment"""
        self.organizer.setup(self.test_dir)
        modelname = self.organizer.modelname
        self.organizer.init('testexp0')
        self.assertTrue(osp.exists(osp.join(
            self.test_dir, modelname, 'experiments', 'testexp0')))
        self.assertIn('testexp0', self.organizer.config.experiments)

        # test without argument
        self.organizer.setup(self.test_dir)
        modelname = self.organizer.modelname
        self.organizer.init()
        self.assertTrue(osp.exists(osp.join(
            self.test_dir, modelname, 'experiments', 'testexp1')))
        self.assertIn('testexp1', self.organizer.config.experiments)

if __name__ == '__main__':
    unittest.main()