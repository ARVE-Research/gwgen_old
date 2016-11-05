# -*- coding: utf-8 -*-
import unittest
import os.path as osp
import numpy as np
import pandas as pd
import _base_testing as bt


class SensitivityAnalysisTest(bt.BaseTest):
    """Test case to test the sensitivity analysis"""

    @property
    def modelname(self):
        return self.organizer.exp_config['sensitivity_analysis']['model']

    def test_setup(self):
        self._test_init()
        self.organizer.sensitivity_analysis(
            setup={}, experiment=self.organizer.experiment)
        modelname = self.modelname
        self.assertTrue(osp.exists(
            self.organizer.config.models[modelname]['root']))

    def test_init(self, sparse=False):
        self.test_setup()
        experiment = self.organizer.experiment
        self.organizer.exp_config['eval_stations'] = self.stations_file
        if not sparse:
            self.organizer.parse_args(
                ('-id {} sens init -nml thresh=13;14,16 '
                 'gp_shape=0.1;0.4,1.1,0.3').format(experiment).split())
            threshs = np.arange(13, 16)
            shapes = np.arange(0.1, 1.1, 0.3)
        else:
            self.organizer.parse_args(
                ('-id {} sens init -nml thresh=13,15 '
                 'gp_shape=0.1,0.5,0.3').format(experiment).split())
            threshs = np.arange(13, 15)
            shapes = np.arange(0.1, 0.5, 0.3)
        shapes, threshs = np.meshgrid(shapes, threshs)
        n = threshs.size
        modelname = self.modelname
        all_exps = self.organizer.config.experiments
        exp_names = [exp_id for exp_id in all_exps
                     if all_exps[exp_id]['model'] == modelname]
        self.assertEqual(len(exp_names), n)
        df = pd.DataFrame.from_dict([
            all_exps[exp_id]['namelist']['weathergen_ctl']
            for exp_id in exp_names])
        self.assertAlmostArrayEqual(df.thresh.values,
                                    threshs.ravel())
        self.assertAlmostArrayEqual(
            df.gp_shape.values, shapes.ravel())

    def test_init_and_param(self):
        self.test_setup()
        experiment = self.organizer.experiment
        self.organizer.exp_config['eval_stations'] = self.stations_file
        self.organizer.exp_config['param_stations'] = self.stations_file
        n_shape = 2
        self.organizer.parse_args(
            ('-id {} sens init -nml thresh=10,16,5 '
             'gp_shape=-1err,1err,{}').format(experiment, n_shape).split())
        modelname = self.modelname
        threshs = np.arange(10, 16, 5)
        _, threshs = np.meshgrid(range(n_shape + 1), threshs)
        n = threshs.size
        all_exps = self.organizer.config.experiments
        exp_names = [exp_id for exp_id in all_exps
                     if all_exps[exp_id]['model'] == modelname]
        self.assertEqual(len(exp_names), n)
        df = pd.DataFrame([
            all_exps[exp_id]['namelist']['weathergen_ctl']
            for exp_id in exp_names], index=exp_names)
        self.assertAlmostArrayEqual(df.thresh.values,
                                    threshs.ravel())
        # get the parameterization values
        df2 = pd.DataFrame([
            all_exps[exp_id].get('parameterization', {}).get('prcp', {})
            for exp_id in exp_names], index=exp_names)[
                ['gpshape_std', 'gpshape_mean']]
        ranges = iter([])
        for i, (exp, err, mean) in enumerate(df2.to_records()):
            if not np.isnan(err):
                ranges = iter(np.linspace(mean - err, mean + err, n_shape))
            else:
                self.assertAlmostEqual(
                    df.gp_shape[i],
                    next(ranges),
                    msg=("Wrong shape parameter for experiment %s.\n"
                         "Namelists: \n%s \n"
                         "Parameterization: \n%s") % (exp, df, df2))

    def test_compile_model(self):
        self.test_setup()
        self.organizer.sensitivity_analysis(
            compile={}, experiment=self.organizer.experiment)
        binpath = osp.join(self.test_dir, self.modelname, 'bin', 'weathergen')
        self.assertTrue(osp.exists(binpath),
                        msg='binary %s does not exist!' % binpath)

    def test_run(self, param=False):
        if param:
            self.test_init_and_param()
        else:
            self.test_init(sparse=True)
        self.organizer.sensitivity_analysis(
            compile={}, run={}, experiment=self.organizer.experiment)
        modelname = self.modelname
        all_exps = self.organizer.config.experiments
        exp_names = [exp_id for exp_id in all_exps
                     if all_exps[exp_id]['model'] == modelname]
        for exp in exp_names:
            self.assertIn(
                'outdata', all_exps[exp],
                msg=('Run failed for experiment %s. No outdata in '
                     'configuration!') % (exp, ))
            self.assertTrue(osp.exists(all_exps[exp]['outdata']),
                            msg='Missing output file %s for experiment %s' % (
                            all_exps[exp]['outdata'], exp))

    def test_evaluate(self, param=False, full=False):
        self.test_run(param=param)
        orig_serial = self.organizer.global_config.get('serial')
        # parallel does not work due to matplotlib
        self.organizer.global_config['serial'] = True
        if not full:
            to_evaluate = ['testexp1_sens1', 'testexp1_sens2']
        else:
            to_evaluate = None
        self.organizer.sensitivity_analysis(
            evaluate={'quants': {'names': ['prcp']}, 'ks': {},
                      'experiments': to_evaluate},
            experiment=self.organizer.experiment)
        self.organizer.global_config['serial'] = orig_serial
        all_exps = self.organizer.config.experiments
        if full:
            modelname = self.modelname
            to_evaluate = [exp_id for exp_id in all_exps
                           if all_exps[exp_id]['model'] == modelname]
        for exp in to_evaluate:
            self.assertIn('quants', all_exps[exp]['evaluation'],
                          msg='No quantile evaluation made for %s' % exp)
            self.assertIn('ks', all_exps[exp]['evaluation'],
                          msg='No ks evaluation made for %s' % exp)

    def test_plot(self):
        """Test whether the plot of a simple sensitivity analysis works"""
        self._test_plot()

    def test_plot_complex(self):
        """Test whether the plot of a complex sensitivity analysis works"""
        self._test_plot(True)

    def _test_plot(self, param=False):
        self.test_evaluate(param=param, full=True)
        plot1d_output = osp.join(self.test_dir, 'plot1d.pdf')
        plot2d_output = osp.join(self.test_dir, 'plot2d.pdf')
        self.organizer.sensitivity_analysis(
            experiment=self.organizer.experiment,
            evaluate={'quality': {}},
            plot={'plot1d': {'plot_output': plot1d_output},
                  'plot2d': {'plot_output': plot2d_output},
                  'names': ['prcp']})
        self.assertTrue(osp.exists(plot1d_output),
                        msg='File %s is missing!' % plot1d_output)
        self.assertTrue(osp.exists(plot2d_output),
                        msg='File %s is missing!' % plot2d_output)


if __name__ == '__main__':
    unittest.main()
