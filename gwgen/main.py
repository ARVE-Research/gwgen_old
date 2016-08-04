from __future__ import print_function, division
import os
import copy
import os.path as osp
import six
import glob
from shutil import copyfile, rmtree
import re
import sys
import datetime as dt
from itertools import groupby, chain
from argparse import ArgumentParser, Namespace
import inspect
import logging
from psyplot.compat.pycompat import OrderedDict
from psyplot.data import safe_list
from gwgen.config import Config, ordered_yaml_dump
import gwgen.utils as utils
from gwgen.utils import docstrings

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO


if six.PY2:
    input = raw_input


class FuncArgParser(ArgumentParser):
    """Subclass of an argument parser that get's parts of the information
    from a given function"""

    _finalized = False

    def __init__(self, *args, **kwargs):
        self.__subparsers = None
        super(FuncArgParser, self).__init__(*args, **kwargs)
        self.__arguments = OrderedDict()
        self.__funcs = []
        self.__main = None

    @staticmethod
    def get_param_doc(doc, param):
        """Get the documentation and datatype for a parameter

        This function returns the documentation and the argument for a
        napoleon like structured docstring `doc`

        Parameters
        ----------
        doc: str
            The base docstring to use
        param: str
            The argument to use

        Returns
        -------
        str
            The documentation of the given `param`
        str
            The datatype of the given `param`"""
        arg_doc = docstrings._keep_params(doc, [param]) or \
            docstrings._keep_types(doc, [param])
        dtype = None
        if arg_doc:
            lines = arg_doc.splitlines()
            arg_doc = '\n'.join(lines[1:])
            param_desc = lines[0].split(':', 1)
            if len(param_desc) > 1:
                dtype = param_desc[1].strip()
        return arg_doc, dtype

    def setup_args(self, func):
        """Add the parameters from the given `func` to the parameter settings
        """
        self.__funcs.append(func)
        args_dict = self.__arguments
        args, varargs, varkw, defaults = inspect.getargspec(func)
        full_doc = inspect.getdoc(func)
        doc = docstrings._get_section(full_doc, 'Parameters') + '\n'
        doc += docstrings._get_section(full_doc, 'Other Parameters')
        doc = doc.rstrip()
        default_min = len(args or []) - len(defaults or [])
        for i, arg in enumerate(args):
            if arg == 'self' or arg in args_dict:
                continue
            arg_doc, dtype = self.get_param_doc(doc, arg)
            args_dict[arg] = d = {'dest': arg, 'short': arg.replace('_', '-'), 
                                  'long': arg.replace('_', '-')}
            if arg_doc:
                d['help'] = arg_doc
                if i >= default_min:
                    d['default'] = defaults[i - default_min]
                else:
                    d['positional'] = True
                if dtype == 'bool' and 'default' in d:
                    d['action'] = 'store_false' if d['default'] else \
                        'store_true'
                elif dtype:
                    d['metavar'] = dtype

    def update_arg(self, arg, if_existent=True, **kwargs):
        """Update the `add_argument` data for the given parameter
        """
        if not if_existent:
            self.__arguments.setdefault(arg, kwargs)
        self.__arguments[arg].update(kwargs)
        
    def pop_arg(self, *args, **kwargs):
        """Delete a previously defined argument from the parser
        """
        return self.__arguments.pop(*args, **kwargs)

    def pop_key(self, arg, key, *args, **kwargs):
        """Delete a previously defined key for the `add_argument`
        """
        return self.__arguments[arg].pop(key, *args, **kwargs)

    def create_arguments(self):
        """Create and add the arguments"""
        ret = []
        if not self._finalized:
            for arg, d in self.__arguments.items():
                try:
                    is_positional = d.pop('positional', False)
                    short = d.pop('short')
                    long_name = d.pop('long', None)
                    if is_positional:
                        short = arg
                        long_name = None
                        d.pop('dest', None)
                    if short == long_name:
                        long_name = None
                    args = [short, long_name] if long_name else [short]
                    if not is_positional:
                        for i, arg in enumerate(args):
                            args[i] = '-' * (i + 1) + arg
                    group = d.pop('group', self)
                    ret.append(group.add_argument(*args, **d))
                except Exception:
                    print('Error while creating argument %s' % arg)
                    raise
        else:
            raise ValueError('Parser has already been finalized!')
        self._finalized = True
        return ret

    def add_subparsers(self, *args, **kwargs):
        """
        Add subparsers to this parser

        Parameters
        ----------
        ``*args, **kwargs``
            As specified by the original
            :meth:`argparse.ArgumentParser.add_subparsers` method
        chain: bool
            Default: False. If True, It is enabled to chain subparsers"""
        chain = kwargs.pop('chain', None)
        ret = super(FuncArgParser, self).add_subparsers(*args, **kwargs)
        if chain:
            self.__subparsers = ret
        return ret

    def parse_known_args(self, args=None, namespace=None):
        def groupargs(arg, currentarg=[None]):
            if arg in commands:
                currentarg[0] = arg
            return currentarg[0]
        if self.__subparsers is not None:
            commands = list(self.__subparsers.choices.keys())
            # get the first argument to make sure that everything works
            if args is None:
                args = sys.argv[1:]
            choices_d = OrderedDict()
            remainders = OrderedDict()
            main_args = []
            cmd = None
            for i, (cmd, subargs) in enumerate(groupby(args, groupargs)):
                if cmd is None:
                    main_args += list(subargs)
                else:
                    choices_d[cmd], remainders[cmd] = super(
                        FuncArgParser, self).parse_known_args(
                            main_args + list(subargs))
            main_ns, remainders[None] = self.__parse_main(main_args)
            for key, val in vars(main_ns).items():
                choices_d[key] = val
            return Namespace(**choices_d), list(chain(*remainders.values()))
        # otherwise, use the default behaviour
        return super(FuncArgParser, self).parse_known_args(args, namespace)

    def __parse_main(self, args):
        """Parse the main arguments only. This is a work around for python 2.7
        because argparse does not allow to parse arguments without subparsers
        """
        if six.PY2:
            self.__subparsers.add_parser("dummy")
            return super(FuncArgParser, self).parse_known_args(
                list(args) + ['dummy'])
        return super(FuncArgParser, self).parse_known_args(args)


class ModelOrganizer(object):
    """
    A class for organizing a model

    This class is indended to have hold the basic functions for organizing a
    model. You can subclass the functions ``setup, init`` to fit to your model.
    When using the model from the command line, you can also use the
    :meth:`setup_parser` method to create the argument parsers"""

    commands = ['setup', 'compile', 'init', 'configure', 'info', 'remove',
                'param', 'run', 'evaluate']

    #: The :class:`gwgen.parser.FuncArgParser` to use for initializing the
    #: model. This attribute is set by the :meth:`setup_parser` method and used
    #: by the `start` method
    parser = None

    #: list of str. The keys describing paths for the model
    paths = ['expdir', 'src', 'data', 'param_stations', 'eval_stations',
             'input', 'outdata', 'nc_file',  'project_file', 'plot_file']

    _modelname = None
    _experiment = None

    @property
    def modelname(self):
        """The name of the model that is currently processed"""
        if self._modelname is None:
            exps = self.config.experiments
            if self._experiment is not None and self._experiment in exps:
                return exps[self._experiment]['model']
            try:
                self._modelname = list(self.config.models.keys())[-1]
            except IndexError:  # no model has yet been created ever
                raise ValueError(
                    "No experiment has yet been created! Please run setup "
                    "before.")
        return self._modelname

    @modelname.setter
    def modelname(self, value):
        if value is not None:
            self._modelname = value

    @property
    def experiment(self):
        """The identifier or the experiment that is currently processed"""
        if self._experiment is None:
            self._experiment = list(self.config.experiments.keys())[-1]
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        if value is not None:
            self._experiment = value

    @property
    def exp_config(self):
        """The configuration settings of the current experiment"""
        return self.config.experiments[self.experiment]

    @property
    def model_config(self):
        """The configuration settings of the current model of the experiment"""
        return self.config.models[self.modelname]

    @property
    def global_config(self):
        """The global configuration settings"""
        return self.config.global_config

    @property
    def logger(self):
        """The logger of this organizer"""
        if self._experiment:
            return logging.getLogger(
                '.'.join([__name__, self.name, self.modelname,
                          self.experiment]))
        elif self._modelname:
            return logging.getLogger(
                '.'.join([__name__, self.name, self.modelname]))
        else:
            return logging.getLogger('.'.join([__name__, self.name]))

    def __init__(self, name):
        """
        Parameters
        ----------
        name: str
            The model name"""
        self.name = name
        self.config = Config(name)

    def _get_next_name(self, old):
        nums = re.findall('\d+', old)
        if not nums:
            raise ValueError(
                "Could not estimate a model name! Please use the modelname"
                " argument to provide a model name.")
        num0 = nums[-1]
        num1 = str(int(num0) + 1)
        return old[::-1].replace(num0[::-1], num1[::-1])[::-1]

    @docstrings.get_sectionsf('ModelOrganizer.main')
    @docstrings.dedent
    def main(self, experiment=None, last=False, new=False,
             verbose=False, verbosity_level=None, match=False):
        """
        The main function for parsing global arguments

        Parameters
        ----------
        experiment: str
            The id of the experiment to use
        last: bool
            If True, the last experiment is used
        new: bool
            If True, a new experiment is created
        verbose: bool
            Increase the verbosity level to DEBUG. See also `verbosity_level`
            for a more specific determination of the verbosity
        verbosity_level: str or int
            The verbosity level to use. Either one of ``'DEBUG', 'INFO',
            'WARNING', 'ERROR'`` or the corresponding integer (see pythons
            logging module)
        match: bool
            If True/set, interprete `experiment` as a regular expression 
            (regex) und use the matching experiment"""
        if match:
            patt = re.compile(experiment)
            matches = list(filter(patt.search, self.config.experiments))
            if len(matches) > 1:
                raise ValueError("Found multiple matches for %s: %s" % (
                    experiment, matches))
            elif len(matches) == 0:
                raise ValueError("No experiment matches %s" % experiment)
            experiment = matches[0]
        if last and self.config.experiments:
            self.experiment = None
        elif new and self.config.experiments:
            self.experiment = self._get_next_name(self.experiment)
        else:
            self._experiment = experiment
        if verbose:
            verbose = logging.DEBUG
        elif verbosity_level:
            if verbosity_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                verbose = getattr(logging, verbosity_level)
            else:
                verbose = int(verbosity_level)
        if verbose:
            logging.getLogger(
                utils.get_toplevel_module(inspect.getmodule(self))).setLevel(
                    verbose)
            
    def _get_main_kwargs(self, kwargs):
        return {key: kwargs.pop(key) for key in list(kwargs)
                if key in inspect.getargspec(self.main)[0]}

    docstrings.keep_params('ModelOrganizer.main.parameters', 'experiment')

    def _modify_main(self, parser):
        to_update = {
            'modelname': dict(short='m'),
            'experiment': dict(short='id', help=docstrings.params[
                'ModelOrganizer.main.parameters.experiment'] +
                '. If the `init` argument is called, the `new` argument is '
                'automatically set. Otherwise, if not specified differently, '
                'the last created experiment is used.'),
            'last': dict(short='l'),
            'new': dict(short='n'),
            'verbose': dict(short='v', action='store_true'),
            'verbosity_level': dict(short='vl'),
            'match': dict(short='E')}
        for key, kwargs in to_update.items():
            try:
                parser.update_arg(key, **kwargs)
            except KeyError:
                pass

    @docstrings.get_sectionsf('FuncArgParser.setup')
    @docstrings.dedent
    def setup(self, root_dir, modelname=None, link=False, **kwargs):
        """
        Perform the initial setup for the model

        Parameters
        ----------
        root_dir: str
            The path to the root directory where the experiments, etc. will
            be stored
        modelname: str
            The name of the model that shall be initialized at `root_dir`. A
            new directory will be created namely ``root_dir + '/' + modelname``
        link: bool
            If set, the source files are linked to the original ones instead
            of copied
        """
        models = self.config.models
        if not models and modelname is None:
            modelname = self.name + '0'
        elif modelname is None:  # try to increment a number in the last used
            modelname = self._get_next_name(self.modelname)
        self.main(**kwargs)
        root_dir = osp.abspath(osp.join(root_dir, modelname))
        models[modelname] = OrderedDict([
            ('root', root_dir), ('timestamps', OrderedDict())])
        models[modelname]['src'] = src_dir = 'src'
        src_dir = osp.join(root_dir, src_dir)
        data_dir = self.config.global_config.get('data',
                                                 osp.join(root_dir, 'data'))
        models[modelname]['data'] = self.relpath(data_dir)
        self.modelname = modelname
        self.logger.info("Initializing model %s", modelname)
        self.logger.debug("    Creating root directory %s", root_dir)
        if not osp.exists(root_dir):
            os.makedirs(root_dir)
        if not osp.exists(src_dir):
            os.makedirs(src_dir)
        module_src = osp.join(osp.dirname(__file__), 'src')
        for f in os.listdir(module_src):
            target = osp.join(src_dir, f)
            if osp.exists(target):
                os.remove(target)
            if link:
                os.symlink(osp.relpath(osp.join(module_src, f), src_dir),
                           target)
            else:
                copyfile(osp.join(module_src, f), target)
        return root_dir

    def _modify_setup(self, parser):
        self._modify_main(parser)

    @docstrings.dedent
    def compile(self, **kwargs):
        """
        Compile the model

        Parameters
        ----------
        ``**kwargs``
            Keyword arguments passed to the :meth:`main` method
        """
        import subprocess as spr
        self.main(**kwargs)
        modelname = self.modelname
        self.logger.info("Compiling %s", modelname)
        mdict = self.config.models[modelname]
        mdict['bindir'] = bin_dir = osp.join(mdict['root'], 'bin')
        mdict['bin'] = osp.join(bin_dir, 'weathergen')
        src_dir = self.abspath(mdict['src'])
        if not os.path.exists(bin_dir):
            self.logger.debug("    Creating bin directory %s", bin_dir)
            os.makedirs(bin_dir)
        for f in os.listdir(src_dir):
            self.logger.debug("    Linking %s...", f)
            target = osp.join(bin_dir, f)
            if osp.exists(target):
                os.remove(target)
            os.symlink(osp.relpath(osp.join(src_dir, f), bin_dir), target)
        spr.call(['make', '-C', bin_dir, 'all'])

    @docstrings.dedent
    def configure(self, global_config=False, model_config=False,
                  update_nml=None, serial=False, nprocs=None,
                  max_stations=None, datadir=None, user=None, host=None,
                  port=None, **kwargs):
        """
        Configure the model and experiments

        Parameters
        ----------
        global_config: bool
            If True/set, the configuration are applied globally (already
            existing and configured experiments are not impacted)
        model_config: bool
            Apply the configuration on the entire model instance instead of
            only the single experiment (already existing and configured
            experiments are not impacted)
        update_nml: str or dict
            A python dict or path to a namelist to use for updating the
            namelist of the model
        serial: bool
            Do the parameterization always serial (i.e. not in parallel on
            multiple processors). Does automatically impact global settings
        nprocs: int or 'all'
            Maximum number of processes to when making the parameterization in
            parallel. Does automatically impact global settings and disables
            `serial`
        max_stations: int
            The maximum number of stations to process in one parameterization
            process. Does automatically impact global settings
        datadir: str
            Path to the data directory to use
        user: str
            The username to use when logging into the database
        host: str
            the host which runs the database server
        port: int
            The port to use to log into the the database
        ``**kwargs``
            Other keywords for the :meth:`main` method or a mapping from
            parameterization task name to yaml configuration files with 
            formatoptions for that task"""
        if global_config:
            d = self.config.global_config
        elif model_config:
            self.main(**kwargs)
            d = self.config.models[self.modelname]
        else:
            d = self.config.experiments[self.experiment]

        if update_nml is not None:
            import f90nml
            with open(update_nml) as f:
                ref_nml = f90nml.read(f)
            nml2use = d.setdefault('namelist', OrderedDict())
            for key, nml in ref_nml.items():
                nml2use.setdefault(key, OrderedDict()).update(dict(nml))
        gconf = self.config.global_config
        if serial:
            gconf['serial'] = True
        elif nprocs:
            nprocs = int(nprocs) if nprocs != 'all' else nprocs
            gconf['serial'] = False
            gconf['nprocs'] = nprocs
        if max_stations:
            gconf['max_stations'] = max_stations
        if datadir:
            datadir = osp.abspath(datadir)
            if global_config:
                d['data'] = datadir
            else:
                self.config.models[self.modelname].setdefault('data', datadir)
        if user is not None:
            gconf['user'] = user
        if port is not None:
            gconf['port'] = port
        if host is not None:
            gconf['host'] = '127.0.0.1'

    def _modify_configure(self, parser):
        parser.update_arg('global_config', short='g', long='globally',
                          dest='global_config')
        parser.update_arg('model_config', short='m', long='model',
                          dest='model_config')
        parser.update_arg('datadir', short='d')
        parser.update_arg('update_nml', short='u')
        parser.update_arg('serial', short='s')
        parser.update_arg('nprocs', short='n')
        parser.update_arg('max_stations', short='max', type=int)

    docstrings.keep_params('ModelOrganizer.main.parameters', 'experiment')

    @docstrings.dedent
    def init(self, modelname=None, description=None, **kwargs):
        """
        Initialize a new experiment

        Parameters
        ----------
        modelname: str
            The name of the model that shall be used. If None, the last one
            created will be used
        description: str
            A short summary of the experiment
        ``**kwargs``
            Keyword arguments passed to the :meth:`main` method

        Notes
        -----
        If the experiment is None, a new experiment will be created
        """
        self.main(**kwargs)
        experiments = self.config.experiments
        experiment = self._experiment
        if experiment is None and not experiments:
            experiment = self.name + '_exp0'
        elif experiment is None:
            experiment = self._get_next_name(self.experiment)
        self.experiment = experiment
        modelname = self.modelname
        self.logger.info("Initializing experiment %s of model %s",
                         experiment, modelname)
        exp_dict = experiments.setdefault(experiment, OrderedDict())
        if description is not None:
            exp_dict['description'] = description
        exp_dict['model'] = modelname
        exp_dict['expdir'] = exp_dir = osp.join('experiments', experiment)
        exp_dir = osp.join(self.config.models[modelname]['root'], exp_dir)
        exp_dict['timestamps'] = OrderedDict()

        if not os.path.exists(exp_dir):
            self.logger.debug("    Creating experiment directory %s", exp_dir)
            os.makedirs(exp_dir)
        return exp_dict

    def _modify_init(self, parser):
        self._modify_main(parser)
        parser.update_arg('description', short='d')

    @docstrings.dedent
    def info(self, exp_path=False, model_path=False, global_path=False,
             config_path=False, complete=False, no_fix=False, on_models=False,
             on_globals=False, modelname=None, **kwargs):
        """
        Print information on the experiments

        Parameters
        ----------
        exp_path: bool
            If True/set, print the filename of the experiment configuration
        model_path: bool
            If True/set, print the filename on the model configuration
        global_path: bool
            If True/set, print the filename on the global configuration
        config_path: bool
            If True/set, print the path to the configuration directory
        complete: bool
            If True/set, the information on all experiments are printed
        no_fix: bool
            If set, paths are given relative to the root directory of the
            model
        on_models: bool
            If set, show information on the models rather than the
            experiment
        on_globals: bool
            If set, show the global configuration settings
        modelname: str
            The name of the model that shall be used. If provided and
            `on_models` is not True, the information on all experiments for
            this model will be shown
        """
        self.main(**kwargs)
        paths = OrderedDict([
            ('conf_dir', config_path), ('_globals_file', global_path),
            ('_model_file', model_path), ('_exp_file', exp_path)])
        if any(paths.values()):
            for key, val in paths.items():
                if val:
                    print(getattr(self.config, key))
            return
        if on_globals:
            complete = True
            no_fix = True
            base = self.config.global_config
        elif on_models:
            base = self.config.models
            current = modelname or self.modelname
        else:
            current = self.experiment
            if modelname is None:
                base = copy.deepcopy(self.config.experiments)
                if not complete:
                    base[current]['id'] = current
                    base[current].move_to_end('id', last=False)
            else:
                base = OrderedDict(filter(lambda t: t[1]['model'] == modelname,
                                          self.config.experiments.items()))
                complete = True
        if not no_fix:
            self.fix_paths(base)
        if not complete:
            base = base[current]
        print(ordered_yaml_dump(base, default_flow_style=False))

    def _modify_info(self, parser):
        self._modify_main(parser)
        parser.update_arg('exp_path', short='ep')
        parser.update_arg('model_path', short='mp')
        parser.update_arg('global_path', short='gp')
        parser.update_arg('config_path', short='cp')
        parser.update_arg('no_fix', short='nf')
        parser.update_arg('complete', short='a', long='all', dest='complete')
        parser.update_arg('on_models', short='M')
        parser.update_arg('on_globals', short='g', long='globally',
                          dest='on_globals')


    @docstrings.get_sectionsf('ModelOrganizer.fix_paths')
    @docstrings.dedent
    def fix_paths(self, d, root=None, model=None):
        """
        Fix the paths in the given dictionary to get absolute paths

        Paramteres
        ----------
        d: dict
            One experiment configuration dictionary
        root: str
            The root path of the model
        model: str
            The model name

        Returns
        -------
        dict
            The modified `d`

        Notes
        -----
        d is modified in place!"""
        root = root or d.get('root')
        model = model or d.get('model')
        paths = self.paths
        for key, val in d.items():
            if isinstance(val, dict):
                d[key] = self.fix_paths(val, root, model)
            elif key in paths:
                val = d[key]
                if isinstance(val, six.string_types) and not osp.isabs(val):
                    d[key] = self.abspath(val, model, root)
                elif (isinstance(safe_list(val)[0], six.string_types) and
                      not osp.isabs(val[0])):
                    for i in range(len(val)):
                        val[i] = self.abspath(val[i], model, root)
        return d
        
    @docstrings.dedent
    def rel_paths(self, d, root=None, model=None):
        """
        Fix the paths in the given dictionary to get relative paths

        Paramteres
        ----------
        %(ModelOrganizer.fix_paths.parameters)s

        Returns
        -------
        dict
            The modified `d`

        Notes
        -----
        d is modified in place!"""
        root = root or d.get('root')
        model = model or d.get('model')
        paths = self.paths
        for key, val in d.items():
            if isinstance(val, dict):
                d[key] = self.rel_paths(val, root, model)
            elif key in paths:
                val = d[key]
                if isinstance(val, six.string_types) and osp.isabs(val):
                    d[key] = self.relpath(val, model, root)
                elif (isinstance(safe_list(val)[0], six.string_types) and
                      osp.isabs(val[0])):
                    for i in range(len(val)):
                        val[i] = self.relpath(val[i], model, root)
        return d

    docstrings.keep_params('ModelOrganizer.main.parameters', 'experiment')

    @docstrings.dedent
    def remove(self, modelname=None, complete=False,
               yes=False, all_models=False, **kwargs):
        """
        Delete an existing experiment and/or modelname

        Parameters
        ----------
        modelname: str
            The name for which the data shall be removed. If True, the
            model will be determined by the experiment. If not None, all
            experiments for the given model will be removed.
        complete: bool
            If set, delete not only the experiments and config files, but also
            all the model files
        yes: bool
            If True/set, do not ask for confirmation
        all_models: bool
            If True/set, all models are removed

        Warnings
        --------
        This will remove the entire folder and all the related informations in
        the configurations!
        """
        self.main(**kwargs)
        if modelname in self.config.models:
            self.modelname = modelname
        all_experiments = self.config.experiments
        models_info = self.config.models
        if all_models:
            experiments = list(all_experiments.keys())
            models = list(models_info.keys())
        elif modelname is not None:
            experiments = [exp for exp, val in all_experiments.items()
                           if val['model'] == self.modelname]
            models = [self.modelname]
        else:
            experiments = [self.experiment]
            models = [self.modelname]
        if not yes:
            if complete:
                msg = ('Are you sure to remove all experiments (%s) and '
                       'directories for the model instances %s?' % (
                           ', '.join(experiments), ', '.join(models)))
            else:
                msg = ('Are you sure to remove the experiments %s' % (
                    ', '.join(experiments)))
            answer = ''
            while answer.lower() not in ['n', 'no', 'y', 'yes']:
                answer = input(msg + '[y/n] ')
            if answer.lower() in ['n', 'no']:
                return
        for exp in experiments:
            self.logger.debug("Removing experiment %s", exp)
            exp_dict = self.fix_paths(all_experiments.pop(exp))
            if osp.exists(exp_dict['expdir']):
                rmtree(exp_dict['expdir'])

        if complete:
            for model in models:
                self.logger.debug("Removing model %s", model)
                modeldir = models_info.pop(model)['root']
                if osp.exists(modeldir):
                    rmtree(modeldir)

    def _modify_remove(self, parser):
        self._modify_main(parser)
        parser.update_arg('complete', short='a', long='all', dest='complete')
        parser.update_arg('yes', short='y')
        parser.update_arg('all_models', short='am')
        parser.update_arg('modelname', const=True, nargs='?', help=(
            'The name for which the data shall be removed. If set without, '
            'argument, the model will be determined by the experiment. If '
            'specified, all experiments for the given model will be removed.'))

    @docstrings.get_sectionsf('ModelOrganizer.param')
    @docstrings.dedent
    def param(self, complete=False, stations=None, other_exp=None,
              setup_from=None, to_db=False, to_csv=False, database=None,
              norun=False, show=False, **kwargs):
        """
        Parameterize the model

        Parameters
        ----------
        stations: str or list of str
            either a list of stations to use for the parameterization or a
            filename containing a 1-row table with stations
        other_exp: str
            Use the parameterization from another experiment instead of
        setup_from: str
            Determine where to get the data from. If `scratch`, the
            data will be calculated from the raw data. If `file`,
            the data will be loaded from a file, if `db`, the data
            will be loaded from a postgres database (Note that the
            `database` argument must be provided!).
        to_db: bool
            Save the data into a postgresql database (Note that the
            `database` argument must be provided!)
        to_csv: bool
            Save the data into a csv file
        %(get_postgres_engine.parameters)s
        norun: bool or list of str
            If True, only the data is set up and the configuration of the
            experiment is not affected. It can be either a list of 
            parameterization tasks or True or ``'all'``
        """
        from gwgen.parameterization import Parameterizer
        import numpy as np
        task_names = [task.name for task in Parameterizer._registry]
        parameterizer_kws = {
            key: vars(val) if isinstance(val, Namespace) else val
            for key, val in kwargs.items() if key in task_names}
        main_kws = {key: val for key, val in kwargs.items()
                    if key not in task_names}
        self.main(**main_kws)
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        param_dir = exp_dict.setdefault(
            'paramdir', osp.join(exp_dict['expdir'], 'parameterization'))
        if not osp.exists(param_dir):
            os.makedirs(param_dir)
        modelname = self.modelname
        logger = self.logger
        logger.info("Parameterizing experiment %s of model %s",
                    experiment, modelname)
        database = database or exp_dict.get('database')
        global_config = self.config.global_config
        # first we check whether everything works with the database
        # We add 'or None' explicitly because otherwise the user would not be
        # able to reset the settings
        user = global_config.get('user') or None
        port = global_config.get('port') or None
        host = global_config.get('host') or '127.0.0.1'
        if database is not None:
            exp_dict['database'] = database
            engine, engine_str = utils.get_postgres_engine(
                database, user, host, port)
        else:
            engine = None
        # setup up the keyword arguments for the parameterization tasks
        for key, d in parameterizer_kws.items():
            if d.get('setup_from') is None:
                d['setup_from'] = setup_from
            if to_csv and not d.get('to_csv'):
                d['to_csv'] = to_csv
            if to_db and not d.get('to_db'):
                d['to_db'] = to_db
            if other_exp and not d.get('other_exp'):
                d['other_exp'] = other_exp
            exp = d.pop('other_exp', experiment) or experiment
            d['config'] = self.fix_paths(self.config.experiments[
                exp])
            if 'database' in d['config'] and (
                    d['setup_from'] in [None, 'db'] or d.get('to_db')):
                d['engine'] = utils.get_postgres_engine(
                    d['config']['database'], host=host, user=user, port=port)[
                        1]
            d['model_config'] = self.config.models[d['config']['model']]
            self._update_model_with_globals(self.fix_paths(d['model_config']))
        if other_exp and stations is None:
            stations = self.fix_paths(
                self.config.experiments[other_exp]).get('param_stations')
        if isinstance(stations, six.string_types):
            stations = [stations]
        if stations is None:
            try:
                stations = np.loadtxt(exp_dict['param_stations'],
                                      dtype='S11', usecols=[0]).astype(np.str_)
            except KeyError:
                raise ValueError('No parameterization stations specified!')
        elif len(stations) == 1 and osp.exists(stations[0]):
            exp_dict['param_stations'] = self.relpath(stations[0])
            stations = np.loadtxt(
                stations[0], dtype='S11', usecols=[0]).astype(np.str_)
        else:
            exp_dict['param_stations'] = self.relpath(
                osp.join(param_dir, 'stations.dat'))
            np.savetxt(exp_dict['param_stations'], stations, fmt='%s')
        kws_to_keep = {'setup_from', 'config', 'model_config', 'to_csv',
                       'to_db', 'engine'}
        global_conf = self.config.global_config
        # choose keywords for data processing
        base_kws = {
            key: {setup_key: val.get(setup_key) for setup_key in kws_to_keep}
            for key, val in parameterizer_kws.items()}
        tasks = Parameterizer.setup_tasks(
            stations, base_kws, global_conf=global_conf, logger=logger)
        # update experiment namelist and configuration
        exp_nml = exp_dict.setdefault('namelist', OrderedDict())
        param_info = exp_dict.setdefault('parameterization', OrderedDict())
        if norun is True or norun == 'all':
            norun = list(parameterizer_kws)
        else:
            norun = safe_list(norun)
        for task in tasks:
            kws = parameterizer_kws[task.name]
            if task.has_run and task.name not in norun:
                run_kws = task.get_run_kws(kws)
                task_nml, task_info = task.run(**run_kws)
                if task_nml:
                    for key, val in task_nml.items():
                        exp_nml.setdefault(key, OrderedDict()).update(val)
                if task_info:
                    param_info[task.name] = task_info
        return tasks

    def _modify_param(self, parser):
        from gwgen.parameterization import Parameterizer
        def norun(s):
            if s is True or s == 'all':
                return True
            try:
                return bool(int(s))
            except TypeError:
                return s.split(',')
        self._modify_main(parser)
        parser.update_arg('setup_from', short='f', long='from',
                          dest='setup_from')
        parser.update_arg('other_exp', short='ido', long='other_id',
                          dest='other_exp')
        parser.update_arg('stations', short='s')
        parser.update_arg('database', short='db')
        parser.update_arg(
            'norun', short='nr', const=True, nargs='?', 
            type=norun, help=(
                'If set without value or "all" or a number different from 0, '
                'the data is set up and the configuration of the '
                'experiment is not affected. Otherwise it can be a comma '
                'separated list of parameterization tasks for which to only '
                'setup the data'), metavar='task1,task2,...')
        doc = docstrings.params['ModelOrganizer.param.parameters']
        setup_from_doc, setup_from_dtype = parser.get_param_doc(
            doc, 'setup_from')
        other_exp_doc, other_exp_dtype = parser.get_param_doc(doc, 'other_exp')
        fname_doc, _ = parser.get_param_doc(doc, 'to_csv')
        dbname_doc, _ = parser.get_param_doc(doc, 'to_db')

        tasks = utils.unique_everseen(
            Parameterizer.sort_by_requirement(Parameterizer._registry[::-1]),
            lambda t: t.name)
        sps = parser.add_subparsers(title='Parameterization tasks', chain=True)
        for task in tasks:
            sp = sps.add_parser(task.name, help=task.summary)
            fname = task._datafile
            dbname = task.dbname
            sp.add_argument(
                '-f', '--from', choices=['scratch', 'file', 'db'],
                help=setup_from_doc, metavar=setup_from_dtype, 
                dest='setup_from')
            sp.add_argument(
                '-ido', '--other_id', help=other_exp_doc,
                metavar=other_exp_dtype)
            if dbname:
                sp.add_argument('-to_db', action='store_true',
                                help=dbname_doc + ' ' + dbname)
            if fname:
                sp.add_argument('-to_csv', action='store_true',
                                help=fname_doc + ' ' + fname)
            if task.has_run:
                sp.setup_args(task.run)
                task._modify_parser(sp)
                sp.create_arguments()
                
    @docstrings.dedent
    def run(self, ifile=None, ofile=None, odir=None, work_dir=None, 
            remove=False, **kwargs):
        """
        Run the model
        
        Parameters
        ----------
        ifile: str
            The path to the input file. If None, it is assumed that it is
            stored in the ``'input'`` key in the experiment configuration
        ofile: str
            The path to the output file.  If None, it is assumed that it is
            stored in the ``'input'`` key in the experiment configuration or
            it will be stored in ``'odir/exp_id.csv'``. The output directory
            ``'odir'`` is determined by the `odir` parameter
        odir: str
            The path to the output directory. If None and not already saved
            in the configuration, it will default to 
            ``'experiment_dir/outdata'``
        work_dir: str
            The path to the work directory where the binaries are copied to.
        remove: bool
            If True, the `work_dir` will be removed if it already exists
            
        Other Parameters
        ----------------
        ``**kwargs``
            Will be passed to the :meth:`main` method
        """
        import subprocess as spr
        import stat
        import f90nml
        logger = self.logger
        self.main(**kwargs)
        exp_config = self.fix_paths(self.exp_config)
        model_config = self.fix_paths(self.model_config)
        experiment = self.experiment
        if not 'compile' in exp_config['timestamps']:
            self.compile(**kwargs)
        logger.info("Running experiment %s of model %s",
                    experiment, self.modelname)
        if ifile is None:
            ifile = exp_config.get('input', self.model_config.get(
                'input',  self.global_config.get('input')))
        if ifile is None:
            raise ValueError("No input file specified!")
        if ofile is None:
            ofile = exp_config.get('outdata')
        if ofile is None:
            ofile = osp.join(
                odir or exp_config.get('outdir', 
                               osp.join(exp_config['expdir'], 'outdata')),
                str(experiment) + '.csv')
        if work_dir is None:
            work_dir = exp_config.get('workdir', 
                                      osp.join(exp_config['expdir'], 'work'))
        exp_config['outdir'] = odir = osp.dirname(ofile)
        exp_config['outdata'] = ofile
        exp_config['input'] = ifile
        exp_config['indir'] = osp.dirname(ifile)
        exp_config['workdir'] = work_dir
        nml = exp_config.get('namelist', {'weathergen_ctl':{}, 'main_ctl': {}})
        for key in ['weathergen_ctl', 'main_ctl']:
            nml.setdefault(key, {})

        if osp.exists(work_dir) and remove:
            rmtree(work_dir)
        elif not osp.exists(work_dir):
            os.makedirs(work_dir)
        if not osp.exists(odir):
            os.makedirs(odir)

        f = model_config['bin']
        bin_dir = model_config['bindir']
        target = osp.join(work_dir, osp.basename(f))
        logger.debug('Copy executable %s to %s', f, target)
        copyfile(f, target)
        for f in glob.glob(osp.join(bin_dir, '*.o')):
            copyfile(f, osp.join(work_dir, osp.basename(f)))
        os.chmod(target, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        logger.debug('    Name list: %s', ordered_yaml_dump(nml))
        with open(osp.join(work_dir, 'weathergen.nml'), 'w') as f:
            f90nml.write(nml, f)
        
        logger.debug('Running model...')
        logger.debug('    input: %s', ifile)
        logger.debug('    output: %s', ofile)
        t = dt.datetime.now()
        commands = 'cd %s && %s %s %s' % (work_dir, target, ifile, ofile)
        logger.debug(commands)
        spr.check_call(commands, stdout=sys.stdout, stderr=sys.stderr, 
                       shell=True)
        err_msg = "Failed to run the model with '%s'!" % commands
        if not osp.exists(ofile):
            raise RuntimeError(
                (err_msg + "Reason: Output %s missing" % (ofile)))
        else:  # check if the file contains more than one line
            with open(ofile) as f:
                f.readline()
                if f.tell() == os.fstat(f.fileno()).st_size:
                    raise RuntimeError(
                        (err_msg + "Reason: Output %s is empty" % (ofile)))
        logger.debug('Done. Time needed: %s', dt.datetime.now() - t)
    
    def _modify_run(self, parser):
        parser.update_arg('ifile', short='i')
        parser.update_arg('ofile', short='o')
        parser.update_arg('odir', short='od')
        parser.update_arg('work_dir', short='wd')
        parser.update_arg('remove', short='r')
                
    @docstrings.dedent
    def evaluate(self, task, stations=None, **kwargs):
        """
        Evaluate the model
        
        Parameters
        ----------
        task: { 'prepare' }
            The job to do. ``'prepare'`` means, that the input data for the
            given `stations` are processed
        stations: str or list of str
            either a list of stations to use for the parameterization or a
            filename containing a 1-row table with stations. If not given, it
            is taken from the configuration settings"""
        from gwgen.evaluation import Evaluator
        import numpy as np
        main_kwargs = self._get_main_kwargs(kwargs)
        self.main(**main_kwargs)
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        eval_dir = osp.join(exp_dict['expdir'], 'evaluation')
        if not osp.exists(eval_dir):
            os.makedirs(eval_dir)
        modelname = self.modelname
        logger = self.logger
        logger.info("Evaluating experiment %s of model %s",
                    experiment, modelname)

        if isinstance(stations, six.string_types):
            stations = [stations]
        if stations is None:
            try:
                stations = np.loadtxt(exp_dict['eval_stations'],
                                      dtype='S11', usecols=[0]).astype(np.str_)
            except KeyError:
                raise ValueError('No evaluation stations specified!')
        elif len(stations) == 1 and osp.exists(stations[0]):
            exp_dict['eval_stations'] = self.relpath(stations[0])
            stations = np.loadtxt(
                stations[0], dtype='S11', usecols=[0]).astype(np.str_)
        else:
            exp_dict['eval_stations'] = self.relpath(
                osp.join(eval_dir, 'stations.dat'))
            np.savetxt(exp_dict['eval_stations'], stations, fmt='%s')
            
        try:
            task_cls = Evaluator.get_task(task)
        except KeyError:
            raise ValueError(
                "Unknown task %s! Possible tasks are {%s}" % (task, ', '.join(
                    Evaluator._registry)))
        task = task_cls(stations, exp_dict, self.fix_paths(self.model_config),
             global_config=self.global_config)
        task_info = task.run(**kwargs)
        if task_info:
            exp_dict.setdefault('evaluation', OrderedDict()).update(
                {task.name: task_info})
        return task
                
    def _modify_evaluate(self, parser):
        from gwgen.evaluation import Evaluator
        parser.pop_arg('task')
        parser.update_arg('stations', short='s')
        sps = parser.add_subparsers(dest='task', title='Evaluation tasks')
        for name, task in Evaluator._registry.items():
            sp = sps.add_parser(task.name, help=task.summary)
            sp.setup_args(task.run)
            task._modify_parser(sp)
            sp.create_arguments()

    def _update_model_with_globals(self, d):
        datadir = self.config.global_config.get('data')
        if datadir and 'data' not in d:
            d['data'] = datadir
        return d

    def abspath(self, path, model=None, root=None):
        """Returns the path from the current working directory

        We only store the paths relative to the root directory of the model.
        This method fixes those path to be applicable from the working
        directory

        Parameters
        ----------
        path: str
            The original path as it is stored in the configuration
        model: str
            The model to use. If None, the :attr:`modelname` attribute is used
        root: str
            If not None, the root directory of the model

        Returns
        -------
        str
            The path as it is accessible from the current working directory"""
        if root is None:
            root = self.config.models[model or self.modelname]['root']
        return osp.join(root, path)

    def relpath(self, path, model=None, root=None):
        """Returns the relative path from the root directory of the model

        We only store the paths relative to the root directory of the model.
        This method gives you this path from a path that is accessible from the
        current working directory

        Parameters
        ----------
        path: str
            The original path accessible from the current working directory
        model: str
            The model to use. If None, the :attr:`modelname` attribute is used
        root: str
            If not None, the root directory of the model

        Returns
        -------
        str
            The path relative from the root directory"""
        if root is None:
            root = self.config.models[model or self.modelname]['root']
        return osp.relpath(path, root)

    def setup_parser(self, parser=None, subparsers=None):
        commands = self.commands

        if subparsers is None:
            if parser is None:
                parser = FuncArgParser(self.name)
            subparsers = parser.add_subparsers(chain=True)

        ret = {}
        for cmd in commands:
            func = getattr(self, cmd)
            ret[cmd] = sp = subparsers.add_parser(
                cmd, help=docstrings.get_summary(func.__doc__ or ''))
            sp.setup_args(func)
            modifier = getattr(self, '_modify_' + cmd, None)
            if modifier is not None:
                modifier(sp)
        parser.setup_args(self.main)
        self._modify_main(parser)
        self.parser = parser
        self.subparsers = ret
        return parser, subparsers, ret

    def start(self):
        def getnspread(attr):
            """Check whether an experiment id is provided and if yes spread it
            to all subcommands"""
            vals = set(
                getattr(ns, attr, None) for ns in namespaces.values()) - {
                    None}
            if len(vals) > 1:
                raise ValueError("Please do only provide one %s!" % attr)
            elif len(vals):
                val = next(iter(vals))
                for ns in namespaces:
                    if hasattr(ns, attr):
                        setattr(ns, attr, val)
                return val
        self.parser.create_arguments()
        for parser in self.subparsers.values():
            parser.create_arguments()
        namespaces = vars(self.parser.parse_args())
        ts = {}
        for cmd in self.commands:
            if cmd in namespaces:
                ns = namespaces[cmd]
                func = getattr(self, cmd or 'main')
                func(**vars(ns))
                ts[cmd] = str(dt.datetime.now())
        exp = self._experiment
        model_parts = {'setup', 'compile'}
        modelname = self._modelname
        if modelname is not None and model_parts.intersection(ts):
            self.config.models[modelname]['timestamps'].update(
                {key: ts[key] for key in model_parts.intersection(ts)})
        if exp is not None and exp in self.config.experiments:
            modelname = self.modelname
            ts.update(self.config.models[modelname]['timestamps'])
            self.config.experiments[exp]['timestamps'].update(ts)
        try:
            exp_config = self.exp_config
        except KeyError:
            pass
        else:
            self.rel_paths(exp_config)
        self.config.save()


def main():
    organizer = ModelOrganizer('gwgen')
    organizer.setup_parser()
    organizer.start()


if __name__ == '__main__':
    main()
