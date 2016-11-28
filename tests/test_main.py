"""Test module for the :mod:`gwgen.main` module"""
import os.path as osp
import unittest
from gwgen.main import docstrings
import _base_testing as bt


class OrganizerTest(bt.BaseTest):
    """Test the :class:`gwgen.main.ModuleOrganizer` class"""

    def test_compile_model(self):
        """Test the compilation of a model"""
        self.organizer.setup(self.test_dir)
        projectname = self.organizer.projectname
        self.organizer.compile_model()
        binpath = osp.join(self.test_dir, projectname, 'bin', 'weathergen')
        self.assertTrue(osp.exists(binpath),
                        msg='binary %s does not exist!' % binpath)

    def test_wind_bias_correction(self):
        """Test gwgen bias wind"""
#        self.organizer.global_config['serial'] = True
        self._test_init()
        self.organizer.parse_args(
            ('evaluate -s %s prepare -to-csv' % self.stations_file).split())
        self.organizer.parse_args(['run'])
#        self.organizer.global_config['nprocs'] = 2
        self.organizer.parse_args('bias wind'.split())
        self.organizer.fix_paths(self.organizer.exp_config)
        ofile = self.organizer.exp_config['postproc']['bias']['wind'][
            'plot_file']
        self.assertTrue(osp.exists(ofile), msg=ofile + ' is missing')


if __name__ == '__main__':
    unittest.main()
