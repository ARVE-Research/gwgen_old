"""Test module for the :mod:`gwgen.main` module"""
import os.path as osp
import unittest
from gwgen.main import FuncArgParser, docstrings
import _base_testing as bt


class OrganizerTest(bt.BaseTest):
    """Test the :class:`gwgen.main.ModuleOrganizer` class"""

    def test_setup(self):
        """Reimplemented to do the test here"""
        self._test_setup()

    def test_init(self):
        """Reimplemented to do the test here"""
        self._test_init()

    def test_compile_model(self):
        """Test the compilation of a model"""
        self.organizer.setup(self.test_dir)
        modelname = self.organizer.modelname
        self.organizer.compile_model()
        binpath = osp.join(self.test_dir, modelname, 'bin', 'weathergen')
        self.assertTrue(osp.exists(binpath),
                        msg='binary %s does not exist!' % binpath)

    def test_set_value(self):
        """Test set_value command"""
        self._test_init()
        self.organizer.parse_args(['set-value', 'test=1', 'test2=test'])
        exp_config = self.organizer.exp_config
        self.assertEqual(exp_config['test'], '1')
        self.assertEqual(exp_config['test2'], 'test')
        self.organizer.parse_args(['set-value', 'testd.okay=12', '-dt', 'int'])
        self.assertEqual(exp_config['testd']['okay'], 12)

    def test_get_value(self):
        """Test get_value command"""
        self.test_set_value()
        self.organizer.print_ = str
        val = self.organizer.parse_args(['get-value', 'testd.okay']).get_value
        self.assertEqual(int(val), self.organizer.exp_config['testd']['okay'])

    def test_del_value(self):
        """Test del_value command"""
        self.test_set_value()
        self.organizer.parse_args(['del-value', 'test'])
        self.assertNotIn('test', self.organizer.exp_config)

    def test_info(self):
        from gwgen.config import ordered_yaml_load
        self._test_init()
        organizer = self.organizer
        organizer.print_ = str
        # test exp_config
        d = ordered_yaml_load(organizer.parse_args(['info', '-nf']).info)
        self.assertEqual(d.pop('id'), organizer.experiment)
        self.assertEqual(d, organizer.exp_config)

        # test model_config
        d = ordered_yaml_load(organizer.parse_args(['info', '-M', '-nf']).info)
        self.assertEqual(d, organizer.model_config)

        # test global config
        d = ordered_yaml_load(organizer.parse_args(['info', '-g', '-nf']).info)
        self.assertEqual(d, organizer.global_config)

        # test all
        organizer.init(new=True)
        d = ordered_yaml_load(organizer.parse_args(['info', '-a', '-nf']).info)
        self.assertEqual(d, organizer.config.experiments)

        # test if the modelname argument works
        modelname = organizer.modelname
        organizer.setup(self.test_dir)  # make a new model the current one
        self.assertNotEqual(modelname, organizer.modelname,
                            msg='Modelnames should differ after setup!')
        # test model_config
        d = ordered_yaml_load(organizer.parse_args(
            ['info', '-M', '-m', modelname, '-nf']).info)
        self.assertEqual(d, organizer.config.models[modelname])




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

    def test_subparser_renamed(self):
        """Test whether one can use the same argument for a subparser again"""
        parser = FuncArgParser()
        parser.add_argument('-a')
        sps = parser.add_subparsers(chain=True)
        # first subparser
        sp1 = sps.add_parser('sp1')
        sp1.add_argument('-t', action='store_false')
        sps1 = sp1.add_subparsers(chain=False)

        # Add a parser with the same name as the following subparser
        sp11_root = sps.add_parser('sp11')
        sp11_root.add_argument('-b')

        # subparsers of first subparser (second level)
        sp11 = sps1.add_parser('sp11')
        sp11.add_argument('-b')
        sp12 = sps1.add_parser('sp12')
        sp12.add_argument('-c', action='store_true')

        # second subparser
        sp2 = sps.add_parser('sp2')
        sp2.add_argument('-t', action='store_false')
        args = parser.parse_args(
            '-a test sp11 -b root sp1 -t sp11 -b okay sp2'.split())
        self.assertEqual(args.sp1.b, 'okay')
        self.assertEqual(args.sp11.b, 'root')

    def test_subparser_renamed_chain(self):
        """Test whether one can use the same argument for a subparser again
        with chain"""
        parser = FuncArgParser()
        parser.add_argument('-a')
        sps = parser.add_subparsers(chain=True)
        # first subparser
        sp1 = sps.add_parser('sp1')
        sp1.add_argument('-t', action='store_false')
        sps1 = sp1.add_subparsers(chain=True)

        # Add a parser with the same name as the following subparser
        sp11_root = sps.add_parser('sp11')
        sp11_root.add_argument('-b')

        # subparsers of first subparser (second level)
        sp11 = sps1.add_parser('sp11')
        sp11.add_argument('-b')
        sp12 = sps1.add_parser('sp12')
        sp12.add_argument('-c', action='store_true')

        # second subparser
        sp2 = sps.add_parser('sp2')
        sp2.add_argument('-t', action='store_false')
        args = parser.parse_args(
            '-a test sp11 -b root sp1 -t sp11 -b okay sp12 -c sp2'.split())
        self.assertEqual(args.sp1.sp11.b, 'okay')
        self.assertEqual(args.sp11.b, 'root')


if __name__ == '__main__':
    unittest.main()
