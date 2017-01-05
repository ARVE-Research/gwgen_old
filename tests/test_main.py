"""Test module for the :mod:`gwgen.main` module"""
import os.path as osp
import unittest
from gwgen.utils import file_len
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

    def test_run(self):
        self._test_init()
        organizer = self.organizer
        exp = organizer.experiment
        organizer.evaluate(self.stations_file, prepare={}, to_csv=True,
                           experiment=exp)
        self.organizer.run()
        fname = osp.join(self.test_dir, organizer.projectname, 'experiments',
                         exp, 'outdata', exp + '.csv')
        self.assertTrue(osp.exists(fname), msg='Output file %s not found!' % (
            fname))
        nlines = file_len(fname)
        self.assertGreater(nlines, 2, msg='No output generated!')

    def test_wind_bias_correction(self):
        """Test gwgen bias wind"""
        self._test_init()
        self.organizer.parse_args(
            ('evaluate -s %s prepare -to-csv' % self.stations_file).split())
        ifile = osp.join(bt.test_root, 'test_data', 'input.csv')
        self.organizer.parse_args(['run',  '-i', ifile])
        self.organizer.parse_args('bias wind'.split())
        self.organizer.fix_paths(self.organizer.exp_config)
        ofile = self.organizer.exp_config['postproc']['bias']['wind'][
            'plot_file']
        self.assertTrue(osp.exists(ofile), msg=ofile + ' is missing')


if __name__ == '__main__':
    unittest.main()
