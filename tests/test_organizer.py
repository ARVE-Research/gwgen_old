import os
import os.path as osp
import shutil
import unittest
from gwgen.main import ModelOrganizer, FuncArgParser, docstrings
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
        binpath = osp.join(self.test_dir, modelname, 'bin', 'weathergen')
        self.assertTrue(osp.exists(binpath),
                        msg='binary %s does not exist!' % binpath)

    def test_init(self):
        """Test the intialization of a new experiment"""
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


class ParserTest(unittest.TestCase):
    """Class to test the :class:`gwgen.main.FuncArgParser` class"""

    def test_positional(self):
        """Test whether positional arguments are captured correctly"""
        def test_positional(name):
            pass
        parser = FuncArgParser()
        dtype = 'str'
        help = 'Just a dummy name'
        doc = docstrings.dedents("""
            Test function for positional argument

            Parameters
            ----------
            name: %s
                %s""" % (dtype, help))
        test_positional.__doc__ = doc
        parser.setup_args(test_positional)
        action = parser.create_arguments()[0]
        self.assertEqual(action.help.strip(), help)
        self.assertEqual(action.metavar, dtype)
        self.assertEqual(action.dest, 'name')
        self.assertTrue(action.required)

    def test_optional(self):
        """Test whether positional arguments are captured correctly"""
        default = 'test'

        def test_optional(name=default):
            pass

        parser = FuncArgParser()
        dtype = 'str'
        help = 'Just a dummy name'
        doc = docstrings.dedents("""
            Test function for positional argument

            Parameters
            ----------
            name: %s
                %s""" % (dtype, help))
        test_optional.__doc__ = doc
        parser.setup_args(test_optional)
        action = parser.create_arguments()[0]
        self.assertEqual(action.help.strip(), help)
        self.assertEqual(action.metavar, dtype)
        self.assertEqual(action.dest, 'name')
        self.assertTrue(action.default, default)

    def test_switch(self):
        """Test whether switches are captured correctly"""
        def test_switch(name=False):
            pass
        parser = FuncArgParser()
        dtype = 'bool'
        help = 'Just a dummy name'
        doc = docstrings.dedents("""
            Test function for positional argument

            Parameters
            ----------
            name: %s
                %s""" % (dtype, help))
        test_switch.__doc__ = doc
        parser.setup_args(test_switch)
        action = parser.create_arguments()[0]
        self.assertEqual(action.help.strip(), help)
        self.assertIsNone(action.metavar)
        self.assertEqual(action.dest, 'name')
        self.assertFalse(action.default)
        self.assertTrue(action.const)

    def test_subparser_chain(self):
        '''Test whether the subparser chaining works'''
        parser = FuncArgParser()
        parser.add_argument('-a')
        sps = parser.add_subparsers(chain=True)
        # first subparser
        sp1 = sps.add_parser('sp1')
        sp1.add_argument('-t', action='store_false')
        sps1 = sp1.add_subparsers(chain=True)

        # subparsers of first subparser (second level)
        sp11 = sps1.add_parser('sp11')
        sp11.add_argument('-b')
        sp12 = sps1.add_parser('sp12')
        sp12.add_argument('-c', action='store_true')

        # second subparser
        sp2 = sps.add_parser('sp2')
        sp2.add_argument('-t', action='store_false')

        args = parser.parse_args(
            '-a test sp1 -t sp11 -b okay sp12 -c sp2'.split())
        # first level test
        self.assertTrue(args.a, 'test')
        self.assertTrue(args.sp1.a, 'test')
        self.assertTrue(args.sp2.a, 'test')
        self.assertFalse(args.sp1.t)
        self.assertTrue(args.sp2.t)

        # second level test
        self.assertFalse(args.sp1.sp11.t)
        self.assertFalse(args.sp1.sp12.t)
        self.assertEqual(args.sp1.sp11.b, 'okay')
        self.assertTrue(args.sp1.sp12.c)

    def test_argument_modification(self):
        """Test the modification of arguments"""

        def test(name='okay', to_delete=True):
            '''
            That's a test function

            Parameters
            ----------
            name: str
                Just a test
            to_delete: bool
                A parameter that will be deleted

            Returns
            -------
            NoneType
                Nothing'''
        parser = FuncArgParser()
        parser.setup_args(test)
        parser.update_arg('name', help='replaced', metavar='replaced')


if __name__ == '__main__':
    unittest.main()
