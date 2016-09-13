from __future__ import print_function, division
import os
import glob
import copy
import os.path as osp
import six
import shutil
import re
import sys
from functools import partial
import datetime as dt
from itertools import groupby, chain, repeat
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
import inspect
import logging
import numpy as np
from psyplot.compat.pycompat import (
    OrderedDict, getcwd, builtins, filterfalse)
from psyplot.data import safe_list
from gwgen.config import Config, ordered_yaml_dump, ordered_yaml_load
import gwgen.utils as utils
from gwgen.utils import docstrings


if six.PY2:
    input = raw_input


class FuncArgParser(ArgumentParser):
    """Subclass of an argument parser that get's parts of the information
    from a given function"""

    _finalized = False

    def __init__(self, *args, **kwargs):
        self._subparsers_action = None
        super(FuncArgParser, self).__init__(*args, **kwargs)
        self.__arguments = OrderedDict()
        self.__funcs = []
        self.__main = None
        self.__currentarg = None
        self.__chain = False

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
        arg_doc = docstrings.keep_params_s(doc, [param]) or \
            docstrings.keep_types_s(doc, [param])
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

    def update_arg(self, arg, if_existent=None, **kwargs):
        """
        Update the `add_argument` data for the given parameter

        Parameters
        ----------
        arg: str
            The name of the function argument
        if_existent: bool or None
            If True, the argument is updated. If None (default), the argument
            is only updated, if it exists. Otherwise, if False, the given
            ``**kwargs`` are only used if the argument is not yet existing
        ``**kwargs``
            The keyword arguments any parameter for the
            :meth:`argparse.ArgumentParser.add_argument` method
        """
        if if_existent or (if_existent is None and arg in self.__arguments):
            self.__arguments[arg].update(kwargs)
        elif not if_existent and if_existent is not None:
            self.__arguments.setdefault(arg, kwargs)

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
                    not_positional = int(not d.pop('positional', False))
                    short = d.pop('short')
                    long_name = d.pop('long', None)
                    if not not_positional:
                        short = arg
                        long_name = None
                        d.pop('dest', None)
                    if short == long_name:
                        long_name = None
                    args = []
                    if short:
                        args.append('-' * not_positional + short)
                    if long_name:
                        args.append('--' * not_positional + long_name)
                    group = d.pop('group', self)
                    if d.get('action') in ['store_true', 'store_false']:
                        d.pop('metavar', None)
                    ret.append(group.add_argument(*args, **d))
                except Exception:
                    print('Error while creating argument %s' % arg)
                    raise
        else:
            raise ValueError('Parser has already been finalized!')
        self._finalized = True
        return ret

    def append2help(self, arg, s):
        """Append the given string to the help of argument `arg`

        Parameters
        ----------
        arg: str
            The function argument
        s: str
            The string to append to the help"""
        self.__arguments[arg]['help'] += s

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
            self.__chain = True
        self._subparsers_action = ret
        return ret

    def grouparg(self, arg, my_arg=None, parent_cmds=[]):
        if self._subparsers_action is None:
            return None
        commands = self._subparsers_action.choices
        currentarg = self.__currentarg
        ret = currentarg or my_arg
        if currentarg is not None:
            sp_key = commands[currentarg].grouparg(arg, currentarg, chain(
                commands, parent_cmds))
            if sp_key is None and arg in commands:
                self.__currentarg = currentarg = arg
                ret = my_arg or currentarg
            elif sp_key not in commands and arg in parent_cmds:
                ret = None
            else:
                ret = my_arg or currentarg
        elif arg in commands:
            self.__currentarg = arg
            ret = arg
        elif arg in parent_cmds:
            ret = None
        return ret

    def parse_known_args(self, args=None, namespace=None):
        if self.__chain:
            # get the first argument to make sure that everything works
            if args is None:
                args = sys.argv[1:]
            choices_d = OrderedDict()
            remainders = OrderedDict()
            main_args = []
            cmd = self.__currentarg = None
            for i, (cmd, subargs) in enumerate(groupby(args, self.grouparg)):
                if cmd is None:
                    main_args += list(subargs)
                else:
                    choices_d[cmd], remainders[cmd] = super(
                        FuncArgParser, self).parse_known_args(
                            main_args + list(subargs))
            main_ns, remainders[None] = self.__parse_main(main_args)
            for key, val in vars(main_ns).items():
                choices_d[key] = val
            self.__currentarg = None
            return Namespace(**choices_d), list(chain(*remainders.values()))
        # otherwise, use the default behaviour
        return super(FuncArgParser, self).parse_known_args(args, namespace)

    def __parse_main(self, args):
        """Parse the main arguments only. This is a work around for python 2.7
        because argparse does not allow to parse arguments without subparsers
        """
        if six.PY2:
            self._subparsers_action.add_parser("dummy")
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

    commands = ['setup', 'compile_model', 'init', 'unarchive', 'configure',
                'set_value', 'get_value', 'del_value', 'info',
                'preproc', 'param', 'run', 'evaluate',
                'sensitivity_analysis', 'archive', 'remove']

    #: mapping from the name of the parser command to the method name. Will be
    #: filled by the :meth:`setup_parser` method
    parser_commands = {'compile_model': 'compile',
                       'sensitivity_analysis': 'sens'}

    #: The :class:`gwgen.parser.FuncArgParser` to use for initializing the
    #: model. This attribute is set by the :meth:`setup_parser` method and used
    #: by the `start` method
    parser = None

    #: list of str. The keys describing paths for the model
    paths = ['expdir', 'src', 'data', 'param_stations', 'eval_stations',
             'indir', 'input', 'outdir', 'outdata', 'nc_file',  'project_file',
             'plot_file', 'reference', 'evaldir', 'paramdir', 'workdir',
             'param_grid', 'grid', 'eval_grid']

    no_modification = False

    print_ = six.print_

    @property
    def logger(self):
        """The logger of this organizer"""
        if self._experiment:
            if not self.is_archived(self.experiment):
                return logging.getLogger(
                    '.'.join([__name__, self.name, self.modelname,
                              self.experiment]))
            return logging.getLogger(
                    '.'.join([__name__, self.name, self.experiment]))
        elif self._modelname:
            return logging.getLogger(
                '.'.join([__name__, self.name, self.modelname]))
        else:
            return logging.getLogger('.'.join([__name__, self.name]))

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

    no_modification = False

    def __init__(self, name, config=None):
        """
        Parameters
        ----------
        name: str
            The model name
        config: iun.config.Config
            The configuration of the organizer"""
        self.name = name
        if config is None:
            config = Config(name)
        self.config = config
        self._parser_set_up = False

    @docstrings.get_sectionsf('ModelOrganizer.start', sections=['Returns'])
    @docstrings.dedent
    def start(self, **kwargs):
        """
        Start the commands of this organizer

        Parameters
        ----------
        ``**kwargs``
            Any keyword from the :attr:`commands` or :attr:`parser_commands`
            attribute

        Returns
        -------
        argparse.Namespace
            The namespace with the commands as given in ``**kwargs`` and the
            return values of the corresponding method"""
        ts = {}
        ret = {}
        info_parts = {'info', 'get-value', 'get_value'}
        for cmd in self.commands:
            parser_cmd = self.parser_commands.get(cmd, cmd)
            if parser_cmd in kwargs or cmd in kwargs:
                kws = kwargs.get(cmd, kwargs.get(parser_cmd))
                if isinstance(kws, Namespace):
                    kws = vars(kws)
                func = getattr(self, cmd or 'main')
                ret[cmd] = func(**kws)
                if cmd not in info_parts:
                    ts[cmd] = str(dt.datetime.now())
        exp = self._experiment
        model_parts = {'setup', 'compile', 'compile_model'}
        modelname = self._modelname
        if modelname is not None and model_parts.intersection(ts):
            self.config.models[modelname]['timestamps'].update(
                {key: ts[key] for key in model_parts.intersection(ts)})
        elif not ts:  # don't make modifications for info
            self.no_modification = True
        if exp is not None and exp in self.config.experiments:
            modelname = self.modelname
            try:
                ts.update(self.config.models[modelname]['timestamps'])
            except KeyError:
                pass
            if not self.is_archived(exp):
                self.config.experiments[exp]['timestamps'].update(ts)
        return Namespace(**ret)

    # -------------------------------------------------------------------------
    # -------------------------------- Main -----------------------------------
    # ------------- Parts corresponding to the main functionality -------------
    # -------------------------------------------------------------------------

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

    @docstrings.get_sectionsf('ModelOrganizer.main')
    @docstrings.dedent
    def main(self, experiment=None, last=False, new=False,
             verbose=False, verbosity_level=None, no_modification=False,
             match=False):
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
        no_modification: bool
            If True/set, no modifications in the configuration files will be
            done
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
            try:
                self.experiment = utils.get_next_name(self.experiment)
            except ValueError:
                raise ValueError(
                    "Could not estimate an experiment id! Please use the "
                    "experiment argument to provide an id.")
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
        self.no_modification = no_modification

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
            'no_modification': dict(short='nm'),
            'match': dict(short='E')}
        for key, kwargs in to_update.items():
            try:
                parser.update_arg(key, **kwargs)
            except KeyError:
                pass

    def _get_main_kwargs(self, kwargs):
        return {key: kwargs.pop(key) for key in list(kwargs)
                if key in inspect.getargspec(self.main)[0]}

    # -------------------------------------------------------------------------
    # --------------------------- Infrastructure ------------------------------
    # ---------- General parts for organizing the model infrastructure --------
    # -------------------------------------------------------------------------

    @docstrings.dedent
    def setup(self, root_dir, modelname=None, link=False, src_model=None,
              **kwargs):
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
        src_model: str
            Another model name to use the source model files from

        Other Parameters
        ----------------
        ``**kwargs``
            Are passed to the :meth:`main` method
        """
        models = self.config.models
        if not models and modelname is None:
            modelname = self.name + '0'
        elif modelname is None:  # try to increment a number in the last used
            try:
                modelname = utils.get_next_name(self.modelname)
            except ValueError:
                raise ValueError(
                    "Could not estimate a model name! Please use the modelname"
                    " argument to provide a model name.")
        self.main(**kwargs)
        root_dir = osp.abspath(osp.join(root_dir, modelname))
        models[modelname] = OrderedDict([
            ('root', root_dir), ('timestamps', OrderedDict())])
        models[modelname]['src'] = src_dir = osp.join(root_dir, 'src')
        src_dir = osp.join(root_dir, src_dir)
        data_dir = self.config.global_config.get('data',
                                                 osp.join(root_dir, 'data'))
        models[modelname]['data'] = data_dir
        self.modelname = modelname
        self.logger.info("Initializing model %s", modelname)
        self.logger.debug("    Creating root directory %s", root_dir)
        if not osp.exists(root_dir):
            os.makedirs(root_dir)
        if not osp.exists(src_dir):
            os.makedirs(src_dir)
        if src_model:
            module_src = self.config.models[src_model]['src']
        else:
            module_src = osp.join(osp.dirname(__file__), 'src')
        for f in os.listdir(module_src):
            target = osp.join(src_dir, f)
            if osp.exists(target):
                os.remove(target)
            if link:
                self._link(osp.join(module_src, f), target)
            else:
                shutil.copyfile(osp.join(module_src, f), target)
        return root_dir

    def _modify_setup(self, parser):
        self._modify_main(parser)
        parser.update_arg('src_model', short='src')

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
            try:
                experiment = utils.get_next_name(self.experiment)
            except ValueError:
                raise ValueError(
                    "Could not estimate an experiment id! Please use the "
                    "experiment argument to provide an id.")
        self.experiment = experiment
        modelname = modelname or self.modelname
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
    def archive(self, odir=None, aname=None, fmt=None, modelname=None,
                experiments=None, current_model=False, no_append=False,
                no_model_paths=False, exclude=None, keep_exp=False,
                rm_model=False, dry_run=False, dereference=False, **kwargs):
        """
        Archive one or more experiments or a model instance

        This method may be used to archive experiments in order to minimize the
        amount of necessary configuration files

        Parameters
        ----------
        odir: str
            The path where to store the archive
        aname: str
            The name of the archive (minus any format-specific extension). If
            None, defaults to the modelname
        fmt: { 'gztar' | 'bztar' | 'tar' | 'zip' }
            The format of the archive. If None, it is tested whether an
            archived with the name specified by `aname` already exists and if
            yes, the format is inferred, otherwise ``'tar'`` is used
        modelname: str
            If provided, the entire model is archived
        experiments: str
            If provided, the given experiments are archived. Note that an error
            is raised if they belong to multiple model instances
        current_model: bool
            If True, `modelname` is set to the current model
        no_append: bool
            It True and the archive already exists, it is deleted
        no_model_paths: bool
            If True, paths outside the experiment directories are neglected
        exclude: list of str
            Filename patterns to ignore (see :func:`glob.fnmatch.fnmatch`)
        keep_exp: bool
            If True, the experiment directories are not removed and no
            modification is made in the configuration
        rm_model: bool
            If True, remove all the model files
        dry_run: bool
            If True, set, do not actually make anything
        dereference: bool
            If set, dereference symbolic links. Note: This is automatically set
            for ``fmt=='zip'``
        """
        fnmatch = glob.fnmatch.fnmatch

        def to_exclude(fname):
            if exclude and (fnmatch(exclude, fname) or
                            fnmatch(exclude, osp.basename(fname))):
                return True

        def do_nothing(path, file_obj):
            return

        def tar_add(path, file_obj):
            file_obj.add(path, self.relpath(path), exclude=to_exclude)

        def zip_add(path, file_obj):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for f in files:
                    abs_file = os.path.join(root, f)
                    if not to_exclude(abs_file):
                        file_obj.write(abs_file, self.relpath(abs_file))

        self.main(**kwargs)
        logger = self.logger
        all_exps = self.config.experiments
        if current_model or modelname is not None:
            if current_model:
                modelname = self.modelname
            experiments = [
                exp for exp, d in self.config.experiments.items()
                if not self.is_archived(exp, 0) and d['model'] == modelname]
            if not experiments:
                raise ValueError(
                    "Could not find any unarchived experiment for %s" % (
                        modelname))
        elif experiments is None:
            experiments = [self.experiment]
        already_archived = list(filter(self.is_archived, experiments))
        if already_archived:
            raise ValueError(
                "The experiments %s have already been archived or are not "
                "existent!" % ', '.join(
                    already_archived))
        if modelname is None:
            modelnames = {all_exps[exp]['model'] for exp in experiments}
            if len(modelnames) > 1:
                raise ValueError(
                    "Experiments belong to multiple model instances: %s" % (
                        ', '.join(modelnames)))
            modelname = next(iter(modelnames))

        self.modelname = modelname
        self.experiment = experiments[-1]

        exps2archive = OrderedDict(
            t for t in all_exps.items() if t[0] in experiments)
        model_config = self.config.models[modelname]

        ext_map, fmt_map = self._archive_extensions()
        if aname is None:
            aname = modelname
        if fmt is None:
            ext, fmt = next(
                (t for t in fmt_map.items() if osp.exists(aname + t[0])),
                ['.tar', 'tar'])
        else:
            ext = fmt_map[fmt]
        if odir is None:
            odir = getcwd()
        archive_name = osp.join(odir, aname + ext)
        exists = osp.exists(archive_name)
        if exists and no_append:
            logger.debug('Removing existing archive %s' % archive_name)
            os.remove(archive_name)
            exists = False
        elif exists and fmt not in ['tar', 'zip']:
            raise ValueError(
                "'Cannot append to %s because this is only possible for 'tar' "
                "and 'zip' extension. Not %s" % (archive_name, fmt))
        logger.info('Archiving to %s', archive_name)

        paths = self._get_all_paths(exps2archive)
        root_dir = self.config.models[modelname]['root']
        check_path = partial(utils.dir_contains, root_dir)
        not_included = OrderedDict([
            (key, list(filterfalse(check_path, safe_list(val))))
            for key, val in paths.items()])

        for key, key_paths in not_included.items():
            for p in key_paths:
                logger.warn(
                    '%s for key %s lies outside the model directory and will '
                    'not be included in the archive!', p, key)
        modes = {'bztar': 'w:bz2', 'gztar': 'w:gz', 'tar': 'w', 'zip': 'w'}
        mode = 'a' if exists else modes[fmt]
        atype = 'zip' if fmt == 'zip' else 'tar'
        if dry_run:
            add_dir = do_nothing
            file_obj = None
        elif atype == 'zip':
            import zipfile
            add_dir = zip_add
            file_obj = zipfile.ZipFile(archive_name, mode)
        else:
            import tarfile
            add_dir = tar_add
            file_obj = tarfile.open(archive_name, mode,
                                    dereference=dereference)
        for exp in experiments:
            exp_dir = exps2archive[exp]['expdir']
            logger.debug('Adding %s', exp_dir)
            add_dir(exp_dir, file_obj)

        now = str(dt.datetime.now())  # current time

        # configuration directory
        config_dir = osp.join(root_dir, '.archived')
        if not dry_run and not osp.exists(config_dir):
            os.makedirs(config_dir)
        for exp in experiments:
            conf_file = osp.join(config_dir, exp + '.yml')
            logger.debug('Store %s experiment config to %s', exp, conf_file)
            if not dry_run:
                exps2archive[exp].setdefault('timestamps', {})
                exps2archive[exp]['timestamps']['archive'] = now
                with open(osp.join(config_dir, exp + '.yml'), 'w') as f:
                    ordered_yaml_dump(exps2archive[exp], f)
        # model configuration file
        conf_file = osp.join(config_dir, '.model.yml')
        logger.debug('Store %s model config to %s', modelname, conf_file)
        if not dry_run:
            if not keep_exp:
                archived_ts = model_config.setdefault('archived', {})
                for exp in experiments:
                    archived_ts[exp] = now
            with open(conf_file, 'w') as f:
                ordered_yaml_dump({modelname: model_config}, f)

        logger.debug('Add %s to archive', config_dir)
        add_dir(config_dir, file_obj)

        if not no_model_paths:
            for dirname in os.listdir(root_dir):
                if osp.basename(dirname) not in ['experiments', '.archived']:
                    logger.debug('Adding %s', osp.join(root_dir, dirname))
                    add_dir(osp.join(root_dir, dirname), file_obj)
        if not keep_exp:
            for exp in experiments:
                exp_dir = exps2archive[exp]['expdir']
                logger.debug('Removing %s', exp_dir)
                if not dry_run:
                    all_exps[exp] = archive_name
                    shutil.rmtree(exp_dir)
        if rm_model:
            logger.debug('Removing %s', root_dir)
            if not dry_run:
                shutil.rmtree(root_dir)
        if not dry_run:
            file_obj.close()

    def _modify_archive(self, parser):
        self._modify_main(parser)
        parser.update_arg('odir', short='d')
        parser.update_arg('aname', short='f')
        parser.update_arg('fmt', choices=['bztar', 'gztar', 'tar', 'zip'])
        parser.update_arg(
            'experiments', short='ids', metavar='exp1,[exp2[,...]]]',
            type=lambda s: s.split(','))
        parser.update_arg('current_model', short='M')
        parser.update_arg('no_append', short='na')
        parser.update_arg('no_model_paths', short='nm')
        parser.update_arg('exclude', short='e')
        parser.update_arg('keep_exp', short='k', long='keep')
        parser.update_arg('rm_model', short='rm')
        parser.update_arg('dry_run', short='n')
        parser.update_arg('dereference', short='L')
        return parser

    @docstrings.dedent
    def unarchive(self, experiments=None, archive=None, complete=False,
                  model_data=False, replace_model_config=False, root=None,
                  modelname=None, fmt=None, force=False, **kwargs):
        """
        Extract archived experiments

        Parameters
        ----------
        experiments: list of str
            The experiments to extract. If None the current experiment is used
        archive: str
            The path to an archive to extract the experiments from. If None,
            we assume that the path to the archive has been stored in the
            configuration when using the :meth:`archive` command
        complete: bool
            If True, archives are extracted completely, not only the experiment
            (implies ``model_data = True``)
        model_data: bool
            If True, the data for the model is extracted as well
        replace_model_config: bool
            If True and the model does already exist in the configuration, it
            is updated with what is stored in the archive
        root: str
            An alternative root directory to use. Otherwise the experiment will
            be exctracted to

            1. the root directory specified in the configuration files
               (if the model exists in it) and `replace_model_config` is False
            2. the root directory as stored in the archive
        modelname: str
            The modelname to use. If None, use the one specified in the archive
        fmt: { 'gztar' | 'bztar' | 'tar' | 'zip' }
            The format of the archive. If None, it is inferred
        force: bool
            If True, force to overwrite the configuration of all experiments
            from what is stored in `archive`. Otherwise, the configuration of
            the experiments in `archive` are only used if missing in the
            current configuration
        """
        def extract_file(path):
            if atype == 'zip':
                return file_obj.open(path)
            else:
                return file_obj.extractfile(path)

        self.main(**kwargs)

        logger = self.logger

        model_config = None

        all_exps = self.config.experiments
        all_models = self.config.models
        # ---- set archive
        # if archive is None, check for the archives listed in `experiments`
        # and raise an error if one has not been archived yet or if they belong
        # to different files
        if archive is None:
            # ---- set experiments
            # if experiments is None, use the current experiment. If complete
            # is True, this will be replaced below
            if experiments is None:
                experiments = [self.experiment]
            archives = list(filter(utils.isstring,
                                   map(self.is_archived, experiments)))
            if len(archives) > 1:
                raise ValueError(
                    'The given experiments belong to multiple archives %s!' % (
                        ', '.join(archives)))
            archive = next(iter(archives))
        elif not complete and experiments is None:
            experiments = [self.experiment]

        logger.info('Unarchiving from %s', archive)

        # --- infer compression type
        ext_map, fmt_map = self._archive_extensions()
        if fmt is None:
            try:
                fmt = next(fmt for ext, fmt in ext_map.items()
                           if archive.endswith(ext))
            except StopIteration:
                raise IOError(
                    "Could not infer archive format of {}! Please specify it "
                    "manually using the `fmt` parameter!".format(archive))

        # if no root directory is specified but a modelname, we take the root
        # directory from the configuration if, and only if, the configuration
        # should not be replaced
        if root is None and modelname is not None and not replace_model_config:
            all_models.get(modelname, {}).get('root')

        # ---- open the archive
        modes = {'bztar': 'r:bz2', 'gztar': 'r:gz', 'tar': 'r', 'zip': 'r'}
        atype = 'zip' if fmt == 'zip' else 'tar'
        if atype == 'tar':
            from tarfile import open as open_file
        else:
            from zipfile import ZipFile as open_file
        file_obj = open_file(archive, modes[fmt])

        # ---- if root is None, get it from the archive
        if root is None:
            with extract_file(osp.join('.archived', '.model.yml')) as fmodel:
                modelname_arc, model_config = next(six.iteritems(
                    ordered_yaml_load(fmodel)))
                # use the modelname in archive only, if nothing is specified
                # here
                modelname = modelname or modelname_arc
            # if the modelname is existent in our configuration and already
            # specified, use this one
            if modelname in self.config.models and not replace_model_config:
                root = self.config.models[modelname].get('root')
            else:
                root = model_config.get('root')
        # if we still don't have it, because it was not specified in the
        # archive or the configuration, raise an error
        if root is None:
            raise ValueError("Could not find a root directory path for the "
                             "model. Please specify manually!")

        logger.info('Root directory for the model: %s', root)

        t = str(dt.datetime.now())  # time at the beginning of extraction

        config_files = []

        def fname_filter(m):
            fname = get_fname(m)
            if (dir_contains('.archived', fname) and
                    not osp.basename(fname).startswith('.')):
                config_files.append(fname)
            return (
                complete or fname in config_files or
                (model_data and not dir_contains('experiments', fname)) or
                any(dir_contains(d, fname) for d in dirs))

        if not complete:
            dirs = [osp.join('experiments', exp) for exp in experiments]
            dirs.append(osp.join('.archived', '.model.yml'))
        dir_contains = partial(utils.dir_contains, exists=False)
        if atype == 'zip':
            def get_fname(m):
                return m
            members = list(filter(fname_filter, file_obj.namelist()))
        else:
            def get_fname(m):
                return m.name
            members = list(filter(fname_filter, file_obj.getmembers()))
        logger.debug('Extracting %s files from archive to %s',
                     len(members), root)
        file_obj.extractall(root, members=members)

        # if the model_config yet not has been red, read it now
        if not model_config:
            with open(osp.join(root, '.archived', '.model.yml')) as fmodel:
                modelname_arc, model_config = next(six.iteritems(
                    ordered_yaml_load(fmodel)))
            modelname = modelname or modelname_arc
        if modelname not in all_models or replace_model_config:
            all_models[modelname] = model_config
        else:
            all_models[modelname]['root'] = root

        # get all experiment names in the archive
        arc_exps = [osp.splitext(osp.basename(f))[0] for f in config_files]
        if complete:
            experiments = arc_exps
        else:
            for exp in filter(lambda exp: exp not in arc_exps, experiments[:]):
                logger.warn('Experiment %s was not found in archive!', exp)
                experiments.remove(exp)
        for exp in experiments:
            if force or self.is_archived(exp):
                with open(osp.join(root, '.archived', exp + '.yml')) as fexp:
                    exp_config = ordered_yaml_load(fexp)
                logger.debug('Update configuration for %s', exp)
                all_exps[exp] = exp_config
            else:
                exp_config = all_exps[exp]
                self.rel_paths(exp_config)
            all_models[modelname].get('archived', {}).pop(exp, None)
            exp_config['model'] = modelname
            exp_config['timestamps']['unarchive'] = t

        logger.debug('Done.')

    def _modify_unarchive(self, parser):
        self._modify_main(parser)
        parser.update_arg(
            'experiments', short='ids', metavar='exp1,[exp2[,...]]]',
            type=lambda s: s.split(','))
        parser.update_arg('archive', short='f', long='file')
        parser.update_arg('complete', short='a', long='all')
        parser.update_arg('model_data', short='md')
        parser.update_arg('replace_model_config', short='M')
        parser.update_arg('root', short='d')
        parser.update_arg('force', short=None)
        return parser

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
                shutil.rmtree(exp_dict['expdir'])

        if complete:
            for model in models:
                self.logger.debug("Removing model %s", model)
                modeldir = models_info.pop(model)['root']
                if osp.exists(modeldir):
                    shutil.rmtree(modeldir)

    def _modify_remove(self, parser):
        self._modify_main(parser)
        parser.update_arg('complete', short='a', long='all', dest='complete')
        parser.update_arg('yes', short='y')
        parser.update_arg('all_models', short='am')
        parser.update_arg('modelname', const=True, nargs='?', help=(
            'The name for which the data shall be removed. If set without, '
            'argument, the model will be determined by the experiment. If '
            'specified, all experiments for the given model will be removed.'))

    @docstrings.dedent
    def compile_model(self, modelname=None, **kwargs):
        """
        Compile the model

        Parameters
        ----------
        modelname: str
            The name of the model. If None, use the last one or the one
            specified by the current experiment
        ``**kwargs``
            Keyword arguments passed to the :meth:`main` method
        """
        import subprocess as spr
        self.main(**kwargs)
        modelname = modelname or self.modelname
        self.modelname = modelname
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
            self._link(osp.join(src_dir, f), target)
        spr.call(['make', '-C', bin_dir, 'all'])

    def _modify_compile_model(self, parser):
        """Does nothing since compile takes no special arguments"""
        self._modify_main(parser)
        return parser

    # -------------------------------------------------------------------------
    # -------------------------- Information ----------------------------------
    # ---------- Parts for getting information from the configuration ---------
    # -------------------------------------------------------------------------

    @docstrings.get_sectionsf('ModelOrganizer.info')
    @docstrings.dedent
    def info(self, exp_path=False, model_path=False, global_path=False,
             config_path=False, complete=False, no_fix=False, on_models=False,
             on_globals=False, modelname=None, return_dict=False,
             insert_id=True, only_keys=False, archives=False, **kwargs):
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
        return_dict: bool
            If True, the dictionary is returned instead of printed
        insert_id: bool
            If True and neither `on_models`, nor `on_globals`, nor `modelname`
            is given, the experiment id is inserted in the dictionary
        only_keys: bool
            If True, only the keys of the given dictionary are printed
        archives: bool
            If True, print the archives and the corresponding experiments for
            the specified model
        """
        self.main(**kwargs)

        def get_archives(model):
            ret = OrderedDict()
            for exp in self.config.models[model].get('archived', {}).keys():
                is_archived = self.is_archived(exp)
                if is_archived:
                    ret.setdefault(is_archived, []).append(exp)
            return ret

        paths = OrderedDict([
            ('conf_dir', config_path), ('_globals_file', global_path),
            ('_model_file', model_path), ('_exp_file', exp_path)])
        if any(paths.values()):
            for key, val in paths.items():
                if val:
                    print(getattr(self.config, key))
            return
        if archives:
            base = OrderedDict()
            current = modelname or self.modelname
            if complete:
                for model in self.config.models.keys():
                    d = get_archives(model)
                    if d:
                        base[model] = d
            else:
                base[current] = get_archives(current)
        elif on_globals:
            complete = True
            no_fix = True
            base = self.config.global_config
        elif on_models:
            base = self.config.models
            current = modelname or self.modelname
        else:
            current = self.experiment
            if modelname is None:
                if insert_id:
                    base = copy.deepcopy(self.config.experiments)
                    if not complete:
                        base[current]['id'] = current
                        if six.PY3:
                            base[current].move_to_end('id', last=False)
                else:
                    base = self.config.experiments
            else:
                base = OrderedDict(filter(
                    lambda t: (not self.is_archived(t[0]) and
                               t[1]['model'] == modelname),
                    self.config.experiments.items()))
                complete = True

        if no_fix and not (archives or on_globals):
            self.rel_paths(base)
        if not complete:
            base = base[current]
        if only_keys:
            base = list(base.keys())
        if not return_dict:
            return self.print_(ordered_yaml_dump(
                base, default_flow_style=False).rstrip())
        else:
            return base

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
        parser.update_arg('only_keys', short='k')
        parser.update_arg('archives', short='arc')
        parser.pop_arg('return_dict')
        parser.pop_arg('insert_id')

    docstrings.keep_params('ModelOrganizer.info.parameters',
                           'complete', 'on_models', 'on_globals', 'modelname')
    # those are far too many, so we store them in another key
    docstrings.params['ModelOrganizer.info.common_params'] = (
        docstrings.params.pop(
            'ModelOrganizer.info.parameters.complete|on_models|'
            'on_globals|modelname'))
    docstrings.keep_params('ModelOrganizer.info.parameters', 'no_fix',
                           'only_keys', 'archives')

    @docstrings.dedent
    def get_value(self, keys, complete=False, on_models=False,
                  on_globals=False, modelname=None, no_fix=False,
                  only_keys=False, base='', return_list=False, archives=False,
                  **kwargs):
        """
        Get one or more values in the configuration

        Parameters
        ----------
        keys: list
            A list of keys to get the values of. %(get_value_note)s
        %(ModelOrganizer.info.common_params)s
        %(ModelOrganizer.info.parameters.no_fix|only_keys|archives)s
        base: str
            A base string that shall be put in front of each key in `values` to
            avoid typing it all the time
        return_list: bool
            If True, the list of values corresponding to `keys` is returned,
            otherwise they are printed separated by a new line to the standard
            output
        """
        def pretty_print(val):
            if isinstance(val, dict):
                if only_keys:
                    val = list(val.keys())
                return ordered_yaml_dump(
                    val, default_flow_style=False).rstrip()
            return str(val)
        config = self.info(complete=complete, on_models=on_models,
                           on_globals=on_globals, modelname=modelname,
                           no_fix=no_fix, return_dict=True, insert_id=False,
                           archives=archives, **kwargs)
        ret = [0] * len(keys)
        for i, key in enumerate(keys):
            if base:
                key = base + key
            key, sub_config = utils.go_through_dict(key, config)
            ret[i] = sub_config[key]
        if return_list:
            return ret
        return self.print_('\n'.join(map(pretty_print, ret)))

    def _modify_get_value(self, parser):
        self._modify_main(parser)
        parser.update_arg('keys', nargs='+', metavar='level0.level1.level...')
        parser.update_arg('complete', short='a', long='all', dest='complete')
        parser.update_arg('on_models', short='M')
        parser.update_arg('on_globals', short='g', long='globally',
                          dest='on_globals')
        parser.update_arg('no_fix', short='nf')
        parser.update_arg('only_keys', short='k')
        parser.update_arg('base', short='b')
        parser.update_arg('archives', short='arc')
        parser.pop_arg('return_list')

    @docstrings.dedent
    def del_value(self, keys, complete=False, on_models=False,
                  on_globals=False, modelname=None, base='', dtype=None,
                  **kwargs):
        """
        Delete a value in the configuration

        Parameters
        ----------
        keys: list
            A list of keys to be deleted. %(get_value_note)s
        %(ModelOrganizer.info.common_params)s
        base: str
            A base string that shall be put in front of each key in `values` to
            avoid typing it all the time
        """
        config = self.info(complete=complete, on_models=on_models,
                           on_globals=on_globals, modelname=modelname,
                           return_dict=True, insert_id=False, **kwargs)
        for key in keys:
            if base:
                key = base + key
            key, sub_config = utils.go_through_dict(key, config)
            del sub_config[key]

    def _modify_del_value(self, parser):
        self._modify_main(parser)
        parser.update_arg('keys', nargs='+', metavar='level0.level1.level...')
        parser.update_arg('complete', short='a', long='all', dest='complete')
        parser.update_arg('on_models', short='M')
        parser.update_arg('on_globals', short='g', long='globally',
                          dest='on_globals')
        parser.update_arg('base', short='b')

    # -------------------------------------------------------------------------
    # -------------------------- Configuration --------------------------------
    # ------------------ Parts for configuring the organizer ------------------
    # -------------------------------------------------------------------------

    @docstrings.dedent
    def configure(self, global_config=False, model_config=False,
                  update_nml=None, serial=False, nprocs=None,
                  max_stations=None, datadir=None, database=None, user=None,
                  host=None, port=None, chunksize=None, **kwargs):
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
        database: str
            The name of a postgres data base to write the data to
        user: str
            The username to use when logging into the database
        host: str
            the host which runs the database server
        port: int
            The port to use to log into the the database
        chunksize: int
            The chunksize to use for the parameterization and evaluation
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
        if database is not None:
            d['database'] = database
        if user is not None:
            gconf['user'] = user
        if port is not None:
            gconf['port'] = port
        if host is not None:
            gconf['host'] = '127.0.0.1'
        if chunksize is not None:
            gconf['chunksize'] = chunksize

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
        parser.update_arg('database', short='db')

    @docstrings.get_sectionsf('ModelOrganizer.set_value')
    @docstrings.dedent
    def set_value(self, items, complete=False, on_models=False,
                  on_globals=False, modelname=None, base='', dtype=None,
                  **kwargs):
        """
        Set a value in the configuration

        Parameters
        ----------
        items: dict
            A dictionary whose keys correspond to the item in the configuration
            and whose values are what shall be inserted. %(get_value_note)s
        %(ModelOrganizer.info.common_params)s
        base: str
            A base string that shall be put in front of each key in `values` to
            avoid typing it all the time
        dtype: str
            The name of the data type or a data type to cast the value to
        """
        config = self.info(complete=complete, on_models=on_models,
                           on_globals=on_globals, modelname=modelname,
                           return_dict=True, insert_id=False, **kwargs)
        if isinstance(dtype, six.string_types):
            dtype = getattr(builtins, dtype)
        elif dtype is None:
            dtype = lambda val: val
        for key, value in six.iteritems(dict(items)):
            if base:
                key = base + key
            key, sub_config = utils.go_through_dict(key, config,
                                                    setdefault=OrderedDict)
            if key in self.paths:
                if isinstance(value, six.string_types):
                    value = osp.abspath(value)
                else:
                    value = list(map(osp.abspath, value))
            sub_config[key] = dtype(value)

    def _modify_set_value(self, parser):
        self._modify_main(parser)
        parser.update_arg(
            'items', nargs='+', type=lambda s: s.split('='),
            metavar='level0.level1.level...=value', help="""
                The key-value pairs to set. If the configuration goes some
                levels deeper, keys may be separated by a ``'.'`` (e.g.
                ``'namelists.weathergen'``). Hence, to insert a  ``','``, it
                must be escaped by a preceeding ``'\'``.""")
        parser.update_arg('complete', short='a', long='all', dest='complete')
        parser.update_arg('on_models', short='M')
        parser.update_arg('on_globals', short='g', long='globally',
                          dest='on_globals')
        parser.update_arg('base', short='b')
        parser.update_arg('dtype', short='dt', choices=dir(builtins))

    # -------------------------------------------------------------------------
    # -------------------------- Preprocessing --------------------------------
    # -------------- Preprocessing functions for the experiment ---------------
    # -------------------------------------------------------------------------

    @property
    def preproc_funcs(self):
        """A mapping from preproc commands to the corresponding function"""
        from gwgen.preproc import CloudGHCNMap
        return {'select': self.select,
                'cloud': self.cloud_preproc,
                'test': self.create_test_sample}

    @docstrings.dedent
    def preproc(self, **kwargs):
        """
        Preprocess the data

        Parameters
        ----------
        ``**kwargs``
            Any keyword from the :attr:`preproc` attribute with kws for the
            corresponding function, or any keyword for the :meth:`main` method
        """
        funcs = self.preproc_funcs
        sp_kws = {key: kwargs.pop(key) for key in set(kwargs).intersection(
            funcs)}
        self.main(**kwargs)
        exp_config = self.fix_paths(self.exp_config)
        outdir = exp_config.setdefault('indir', osp.join(
            exp_config['expdir'], 'input'))
        if not osp.exists(outdir):
            os.makedirs(outdir)

        preproc_config = exp_config.setdefault('preproc', OrderedDict())

        for key, val in sp_kws.items():
            if isinstance(val, Namespace):
                val = vars(val)
            info = funcs[key](**val)
            if info:
                preproc_config[key] = info

    def _modify_preproc(self, parser):
        from gwgen.preproc import CloudPreproc
        self._modify_main(parser)
        sps = parser.add_subparsers(title='Preprocessing tasks', chain=True)

        # select
        sp = sps.add_parser(
            'select', help='Select stations based upon a regular grid')
        sp.setup_args(self.select)
        sp.update_arg('grid', short='g')
        sp.update_arg('grid_output', short='og')
        sp.update_arg('stations_output', short='os')
        sp.update_arg('igrid_key', short='k')
        sp.update_arg('grid_key', short='ok')
        sp.update_arg('grid_db', short='gdb')
        sp.update_arg('stations_db', short='sdb')
        sp.update_arg('no_prcp_check', short='nc')
        sp.update_arg('setup_from', short='f', long='from',
                      dest='setup_from')
        sp.update_arg('download', short='d', choices=['single', 'all'])
        sp.create_arguments()

        # cloud preprocessing
        sp = sps.add_parser('cloud', help='Cloud preprocessing')
        sp.setup_args(self.cloud_preproc)
        sp.update_arg('max_files', short='mf', type=int)
        sp.pop_arg('return_manager')
        self._modify_task_parser(sp, CloudPreproc)
        sp.create_arguments()

        # test samples
        sp = sps.add_parser(
            'test', help='Create a test sample for selected GHCN stations')
        sp.setup_args(self.create_test_sample)
        sp.update_arg('no_cloud', short='nc')
        sp.update_arg('reduce_eecra', short='re', type=float)
        sp.update_arg('keep_all', short='a')
        sp.create_arguments()

        return parser

    # ------------------------------- Selection -------------------------------

    def _prcp_check(self, series):
        try:
            return 11 == len(series.to_frame().set_index('prcp').join(
                self._prcp_test, how='inner').prcp.unique())
        except:
            return None

    def _select_best_df(self, df, test_series, kws):
        from gwgen.parameterization import DailyGHCNData
        # disable logging for the DailyGHCNData task
        task_logger = DailyGHCNData([], self.exp_config, self.model_config,
                                    self.global_config).logger
        orig_level = task_logger.level
        task_logger.setLevel(logging.WARNING)
        self._test_series = test_series
        self._select_kws = kws
        self._select_task = DailyGHCNData
        g = df.sort_values('nyrs', ascending=False).groupby(
            level=['clon', 'clat'])
        ret = g.id.agg(self._select_best)
        task_logger.setLevel(orig_level)
        return ret

    def _select_best(self, series):
        test_series = self._test_series
        for station in series.values:
            task = self._select_task(
                np.array([station]), self.exp_config, self.model_config,
                self.global_config, **self._select_kws)
            try:
                task.init_task()
            except FileNotFoundError as e:
                task.logger.warn(e)
            else:
                task.setup()
                if len(test_series) == len(
                    task.data.set_index('prcp').join(
                        test_series, how='inner').prcp.unique()):
                    return station
        return series.values[0]

    @staticmethod
    def _parallel_select(l):
        organizer, df, test_series, kws = l
        return organizer._select_best_df(df, test_series, kws)

    @docstrings.dedent
    def select(self, grid=None, grid_output=None, stations_output=None,
               igrid_key=None, grid_key=None, grid_db=None, stations_db=None,
               no_prcp_check=False, setup_from=None, download=None, **kwargs):
        """
        Select stations based upon a regular grid

        Parameters
        ----------
        grid: str
            The path to a csv-file containing a lat and a lon column with the
            information on the centers of the grid. If None, `igrid_key` must
            not be None and point to a key in the configuration (either the one
            of the experiment, or the model, or the global configuration)
            specifying the path
        grid_output: str
            The path to the csv-file where to store the mapping from grid
            lat-lon to station id.
        stations_output: str
            The path to the csv-file where to store the mapping from station
            to grid center point
        igrid_key: str
            The key in the configuration where to store the path of the `grid`
            input file
        grid_key: str
            The key in the configuration where to store the name of the
            `grid_output` file.
        grid_db: str
            The name of a data table to store the data of `stations_output` in
        stations_db: str
            The name of a data table to store the data for `stations_output` in
        no_prcp_check: bool
            If True, we will not check for the values between 0.1 and 1.0 for
            precipitation and save the result in the ``'best'`` column
        setup_from: { 'scratch' | 'file' | 'db' }
            The setup method for the daily data for the prcp check
        download: { 'single' | 'all' }
            Handles how to manage missing files for the prcp check. If None
            (default), an warning is printed and the file is ignored, if
            ``'single'``, the missing file is downloaded, if ``'all'``, the
            entire tarball is downloaded (strongly not recommended for this
            function)

        Other Parameters
        ----------------
        ``**kwargs``
            are passed to the :meth:`main` method

        Notes
        -----
        for `igrid_key` and `ogrid_key` we recommend one of
        ``{'grid', 'param_grid', 'eval_grid'`` because that implies a
        correct path management
        """
        from gwgen.evaluation import EvaluationPreparation
        import numpy as np
        import scipy.spatial
        import pandas as pd

        logger = self.logger

        if grid is None:
            if igrid_key is not None:
                grid = self.exp_config.get(igrid_key, self.model_config.get(
                    igrid_key, self.global_config.get(igrid_key)))
            else:
                raise ValueError(
                    "No grid file or configuration key specified!")
        if grid is None:
            raise ValueError(
                    "No grid file specified and '%s' could not be found in "
                    "the configuration!" % igrid_key)
        t = EvaluationPreparation(np.array([]), self.exp_config,
                                  self.model_config, self.global_config)
        # get inventory
        t.download_src()
        df_stations = t.station_list
        df_stations = df_stations[df_stations.vname == 'PRCP'].drop(
            'vname', 1).reset_index()  # reset_index required due to filtering
        df_stations['nyrs'] = df_stations.lastyr - df_stations.firstyr

        # read 1D grid information
        df_centers = pd.read_csv(grid)
        df_centers.rename(columns={'lon': 'clon', 'lat': 'clat'}, inplace=True)

        # concatenate lat and lon values into x-y points
        center_points = np.dstack(
            [df_centers.clat.values, df_centers.clon.values])[0]
        station_points = np.dstack([df_stations.lat, df_stations.lon])[0]

        # look up the nearest neighbor
        logger.debug('Searching neighbors...')
        kdtree = scipy.spatial.cKDTree(center_points)
        dist, indexes = kdtree.query(station_points)
        logger.debug('Done.')

        # store the lat and longitude of, and the distance to the center grid
        # point in the stations table
        df_stations['clon'] = df_centers.clon.values[indexes]
        df_stations['clat'] = df_centers.clat.values[indexes]
        df_stations['dist'] = dist

        # --------- stations with the closest distance to grid center ---------
        # group by the center coordinates and look for the index with the
        # smallest distance
        g = df_stations.sort_index().groupby(['clon', 'clat'])
        indices_closest = g.dist.idxmin()
        indices_longest = g.nyrs.idxmax()
        # merge the nearest stations into the centers table
        df_centers.set_index(['clon', 'clat'], inplace=True)
        df_stations.set_index(['clon', 'clat'], inplace=True)
        merged = df_centers.merge(
            df_stations.ix[indices_closest][['id']].rename(
                columns={'id': 'nearest_station'}),
            left_index=True, right_index=True, how='outer')
        merged = merged.merge(
            df_stations.ix[indices_longest][['id']].rename(
                columns={'id': 'longest_record'}),
            left_index=True, right_index=True, how='outer')

        if not no_prcp_check:
            test_series = pd.Series(
                np.arange(0.1, 1.05, 0.1), name='prcp')
            logger.debug('Performing best station check with %s',
                         test_series.values)
            kws = dict(download=download, setup_from=setup_from)
            if not self.global_config.get('serial'):
                import multiprocessing as mp
                nprocs = self.global_config.get('nprocs', 'all')
                lonlats = np.unique(df_stations.dropna(0).index.values)
                if nprocs == 'all':
                    nprocs = mp.cpu_count()
                splitted = np.array_split(lonlats, nprocs)
                try:
                    nprocs = list(map(len, splitted)).index(0)
                except ValueError:
                    pass
                else:
                    splitted = splitted[:nprocs]
                dfs = [df_stations.loc[list(arr)] for arr in splitted]
                # initializing pool
                logger.debug('Start %i processes', nprocs)
                pool = mp.Pool(nprocs)
                args = list(zip(repeat(self), dfs, repeat(test_series),
                                repeat(kws)))
                res = pool.map_async(self._parallel_select, args)
                best = pd.concat(res.get())
                pool.close()
                pool.terminate()
            else:
                best = self._select_best_df(
                    df_stations.dropna(0), test_series, kws)
            merged = merged.merge(
                best.to_frame().rename(columns={'id': 'best'}),
                left_index=True, right_index=True, how='outer')

        if igrid_key:
            self.exp_config[igrid_key] = grid
        if stations_output:
            logger.debug('Dumping to%s %s',
                         ' exisiting' if osp.exists(stations_output) else '',
                         stations_output)
            utils.safe_csv_append(df_stations, stations_output)

        if grid_output:
            logger.debug('Dumping to%s %s',
                         ' exisiting' if osp.exists(grid_output) else '',
                         grid_output)
            utils.safe_csv_append(merged, grid_output)
            if grid_key is not None:
                self.exp_config[grid_key] = grid_output
        if stations_db or grid_db:
            conn = t.engine.connect()
            if stations_db:
                logger.info('Writing %i lines into %s', len(df_stations),
                            stations_db)
                df_stations.to_sql(stations_db, conn, if_exists='append')
            if grid_db:
                logger.info('Writing %i lines into %s', len(merged),
                            grid_db)
                merged.to_sql(grid_db, conn, if_exists='append')
            conn.close()

        return df_stations, merged

    # --------------------------- Cloud inventory -----------------------------

    @docstrings.dedent
    def cloud_preproc(self, max_files=None, return_manager=False, **kwargs):
        """
        Extract the inventory of EECRA stations

        Parameters
        ----------
        max_files: int
            The maximum number of files to process during one process. If None,
            it is determined by the global ``'max_stations'`` key
        ``**kwargs``
            Any task in the :class:`gwgen.preproc.CloudPreproc` framework
        """
        from gwgen.preproc import CloudPreproc
        from gwgen.parameterization import HourlyCloud
        stations_orig = self.global_config.get('max_stations')
        if max_files is not None:
            self.global_config['max_stations'] = max_files
        files = HourlyCloud.from_organizer(self, []).raw_src_files
        manager = CloudPreproc.get_manager(config=self.global_config)
        for key, val in kwargs.items():
            if isinstance(val, Namespace):
                kwargs[key] = val = vars(val)
                val.pop('max_files', None)
        self._setup_manager(manager, stations=list(files.values()),
                            base_kws=kwargs)
        d = {}
        manager.run(d)
        if stations_orig:
            self.global_config['max_stations'] = stations_orig
        else:
            self.global_config.pop('max_stations', None)
        if return_manager:
            return d, manager
        else:
            return d

    # --------------------------- Parameterization ----------------------------

    @docstrings.get_sectionsf('ModelOrganizer.param')
    @docstrings.dedent
    def param(self, complete=False, stations=None, other_exp=None,
              setup_from=None, to_db=None, to_csv=None, database=None,
              norun=False, to_return=None, **kwargs):
        """
        Parameterize the model

        Parameters
        ----------
        stations: str or list of str
            either a list of stations to use or a filename containing a
            1-row table with stations
        other_exp: str
            Use the configuration from another experiment
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
        database: str
            The name of a postgres data base to write the data to
        norun: bool, list of str or ``'all'``
            If True, only the data is set up and the configuration of the
            experiment is not affected. It can be either a list of  tasks or
            True or ``'all'``
        to_return: list of str or ``'all'``
            The names of the tasks to return. If None, only the ones with an
            :attr:`gwgen.utils.TaskBase.has_run` are returned.
        complete: bool
            If True, setup and run all possible tasks
        """
        from gwgen.parameterization import Parameterizer
        task_names = [task.name for task in Parameterizer._registry]
        parameterizer_kws = {
            key: vars(val) if isinstance(val, Namespace) else dict(val)
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
        stations = self._get_stations(stations, other_exp, param_dir,
                                      'param_stations')
        global_conf = self.config.global_config
        # choose keywords for data processing
        manager = Parameterizer.get_manager(config=global_conf)
        self._setup_manager(manager, stations, other_exp, setup_from, to_db,
                            to_csv, database, to_return, complete,
                            parameterizer_kws)
        # update experiment namelist and configuration
        if not norun:
            manager.run(exp_dict.setdefault('parameterization', OrderedDict()),
                        exp_dict.setdefault('namelist', OrderedDict()))
        return manager

    def _modify_param(self, parser, *args, **kwargs):
        from gwgen.parameterization import Parameterizer
        self._modify_task_parser(parser, Parameterizer, *args, **kwargs)

    # --------------------------------- Test ----------------------------------

    @docstrings.dedent
    def create_test_sample(self, test_dir, stations, no_cloud=False,
                           reduce_eecra=0, keep_all=False):
        """
        Create a test sample for the given GHCN stations

        Parameters
        ----------
        test_dir: str
            The path to the directory containing the test files from Github
        stations: str or list of str
            either a list of GHCN stations to use or a filename containing a
            1-row table with GHCN stations
        no_cloud: bool
            If True, no cloud stations are extracted
        reduce_eecra: float
            The percentage by which to reduce the EECRA data
        keep_all: bool
            If True all years of the EECRA data are used. Otherwise, only the
            years with complete temperature and cloud are kept. Note
            that this has only an effect if `reduce_eecra` is not 0
        """
        import calendar
        import pandas as pd
        from gwgen.parameterization import DailyGHCNData, HourlyCloud

        def is_complete(s):
            ndays = 366 if calendar.isleap(s.name[1]) else 365
            s[:] = s.ix[~s.index.duplicated()].count() == ndays
            return s

        stations = self._get_stations(stations)
        np.savetxt(osp.join(test_dir, 'test_stations.dat'), stations, fmt='%s')
        # download the GHCN data
        ghcn_task = DailyGHCNData.from_organizer(self, stations,
                                                 download='single')
        ghcn_task.init_from_scratch()
        data_dir = super(DailyGHCNData, ghcn_task).data_dir

        if not no_cloud:
            eecra_task = HourlyCloud.from_organizer(self, stations)
            if len(eecra_task.stations) == 0:
                raise ValueError(
                    "Could not find any station in the given stations %s!",
                    ', '.join(stations))
            np.savetxt(osp.join(test_dir, 'eecra_test_stations.dat'),
                       eecra_task.eecra_stations, fmt='%i')
            eecra_task.init_from_scratch()

        for fname in ghcn_task.raw_src_files:
            target = fname.replace(osp.join(data_dir, ''),
                                   osp.join(test_dir, ''))
            if not osp.samefile(fname, target):
                shutil.copyfile(fname, target)
            shutil.make_archive(osp.join(test_dir, 'ghcn', 'ghcnd_all'),
                                'gztar',
                                root_dir=osp.join(test_dir, 'ghcn'),
                                base_dir='ghcnd_all')

        if not no_cloud:
            for fname in eecra_task.src_files:
                target = fname.replace(osp.join(data_dir, ''),
                                       osp.join(test_dir, ''))
                if not reduce_eecra and not osp.samefile(fname, target):
                    shutil.copyfile(fname, target)
                else:
                    df = pd.read_csv(fname)
                    if not keep_all:
                        df_bool = df.set_index(
                            ['station_id', 'year', 'month', 'day'])[[
                                'ww', 'AT', 'N']]
                        for col in df_bool.columns:
                            df_bool[col] = df_bool[col].astype(bool)
                        g = df_bool.groupby(level=['station_id', 'year'])
                        mask = g.transform(is_complete).values.any(axis=1)
                        df = df.ix[mask]

                    g = df.groupby(['station_id', 'year'],
                                   as_index=False)
                    tot = g.ngroups
                    n = np.ceil(tot * (100 - reduce_eecra) / 100)
                    idx_groups = iter(sorted(np.random.permutation(tot)[:n]))
                    self.logger.debug(
                        'Saving EECRA test sample with %i years from %i to '
                        '%s', n, tot, target)
                    df.ix[1:0].to_csv(target, index=False)
                    igrp = next(idx_groups)
                    for i, (key, group) in enumerate(g):
                        if i == igrp:
                            group.to_csv(target, header=False, mode='a',
                                         index=False)
                            igrp = next(idx_groups, -1)
    # -------------------------------------------------------------------------
    # ------------------------------- Run -------------------------------------
    # --------------------------- Run the experiment --------------------------
    # -------------------------------------------------------------------------

    @docstrings.get_sectionsf('ModelOrganizer.run')
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
        if not {'compile_model', 'compile'} & set(model_config['timestamps']):
            self.compile_model(**kwargs)
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
                odir or exp_config.get(
                    'outdir', osp.join(exp_config['expdir'], 'outdata')),
                str(experiment) + '.csv')
        if work_dir is None:
            work_dir = exp_config.get('workdir',
                                      osp.join(exp_config['expdir'], 'work'))
        exp_config['outdir'] = odir = osp.dirname(ofile)
        exp_config['outdata'] = ofile
        exp_config['input'] = ifile
        exp_config['indir'] = osp.dirname(ifile)
        exp_config['workdir'] = work_dir
        nml = exp_config.get('namelist',
                             {'weathergen_ctl': {}, 'main_ctl': {}})
        for key in ['weathergen_ctl', 'main_ctl']:
            nml.setdefault(key, {})

        if osp.exists(work_dir) and remove:
            shutil.rmtree(work_dir)
        elif not osp.exists(work_dir):
            os.makedirs(work_dir)
        if not osp.exists(odir):
            os.makedirs(odir)

        f = model_config['bin']
        target = osp.join(work_dir, osp.basename(f))
        logger.debug('Copy executable %s to %s', f, target)
        shutil.copyfile(f, target)
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

    # -------------------------------------------------------------------------
    # -------------------------- Postprocessing -------------------------------
    # ------------ Postprocessing functions for the experiment ----------------
    # -------------------------------------------------------------------------

    # ---------------------------- Evaluation ---------------------------------

    @docstrings.dedent
    def evaluate(self, stations=None, other_exp=None,
                 setup_from=None, to_db=None, to_csv=None, database=None,
                 norun=False, to_return=None, complete=False, **kwargs):
        """
        Evaluate the model

        Parameters
        ----------
        %(ModelOrganizer.param.parameters)s"""
        from gwgen.evaluation import Evaluator
        task_names = [task.name for task in Evaluator._registry]
        evaluator_kws = {
            key: vars(val) if isinstance(val, Namespace) else dict(val)
            for key, val in kwargs.items() if key in task_names}
        main_kws = {key: val for key, val in kwargs.items()
                    if key not in task_names}
        self.main(**main_kws)
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        eval_dir = exp_dict.setdefault(
            'evaldir', osp.join(exp_dict['expdir'], 'evaluation'))
        if not osp.exists(eval_dir):
            os.makedirs(eval_dir)
        modelname = self.modelname
        logger = self.logger
        logger.info("Evaluating experiment %s of model %s",
                    experiment, modelname)
        stations = self._get_stations(stations, other_exp, eval_dir,
                                      'eval_stations')
        global_conf = self.config.global_config
        # choose keywords for data processing
        manager = Evaluator.get_manager(config=global_conf)
        self._setup_manager(manager, stations, other_exp, setup_from, to_db,
                            to_csv, database, to_return, complete,
                            evaluator_kws)
        # update experiment namelist and configuration
        if not norun:
            manager.run(exp_dict.setdefault('evaluation', OrderedDict()))
        return manager

    def _modify_evaluate(self, parser, *args, **kwargs):
        from gwgen.evaluation import Evaluator
        self._modify_task_parser(parser, Evaluator, *args, **kwargs)

    # ----------------------- Sensitivity analysis ----------------------------

    @docstrings.dedent
    def sensitivity_analysis(self, **kwargs):
        """
        Perform a sensitivity analysis on the given parameters

        This function performs a sensitivity analysis on the current
        experiment. It creates a new model and uses the evaluation and
        parameterization of the current model to get information on the others
        """
        from gwgen.sensitivity_analysis import SensitivityAnalysis
        sa_func_map = OrderedDict([
            ('setup', 'setup'), ('compile', 'compile_model'),
            ('init', 'init'), ('run', 'run'), ('evaluate', 'evaluate'),
            ('plot', 'plot'), ('remove', 'remove')])
        sensitivity_kws = OrderedDict(
            (key, kwargs[key]) for key in sa_func_map if key in kwargs)
        main_kws = {
            key: kwargs[key] for key in set(kwargs).difference(sa_func_map)}
        self.main(**main_kws)
        # to make sure, we already called the choose the right experiment and
        # modelname
        experiment = self.experiment
        sa = SensitivityAnalysis(self)
        self.fix_paths(self.exp_config)
        self.fix_paths(self.model_config)
        for key, val in sensitivity_kws.items():
            if isinstance(val, Namespace):
                val = vars(val)
            getattr(sa, sa_func_map[key])(**val)

    def _modify_sensitivity_analysis(self, parser):
        from gwgen.sensitivity_analysis import (
            SensitivityAnalysis, SensitivityPlot)

        def params_type(s):
            splitted = s.split('=', 1)
            key = splitted[0]
            return key, utils.str_ranges(splitted[1])

        sps = parser.add_subparsers(help='Sensitivity analysis subroutines',
                                    chain=True)

        # setup parser
        sp = sps.add_parser('setup',
                            help='Setup the sensitivity analysis model')
        sp.setup_args(SensitivityAnalysis.setup)
        self._modify_main(sp)
        sp.update_arg('no_move', short='nm')
        sp.create_arguments()

        # compile parser
        sp = sps.add_parser('compile',
                            help='Compile the sensitivity analysis model')
        sp.setup_args(SensitivityAnalysis.compile_model)
        self._modify_compile_model(sp)
        sp.create_arguments()

        # init parser
        sp = sps.add_parser(
            'init', help='Initialize the sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.init)
        sp.update_arg('experiment', short='id')
        sp.update_arg('nml', long='namelist', type=params_type, help="""
            A list from namelist parameters and their values to use. Range
            parts might be lists of numbers or ``'<i>err'`` to use
            ``'<i>'``-times the error from the parameterization. You might also
            provide up to three values in case on of them is a string with
            ``'err'`` in it, where the first value corresponds to the minimum,
            the second to  the maximum and the third to the number of steps.
            """, metavar='nml_key=f1[,f2[,f3]]', nargs='+')
        sp.update_arg('run_prepare', short='prep')
        sp.update_arg('no_move', short='nm')
        sp.create_arguments()

        # run parser
        sp = sps.add_parser(
            'run', help='Run the sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.run)
        sp.update_arg('remove', short='rm')
        sp.create_arguments()

        # evaluate parser
        sp = sps.add_parser(
            'evaluate', help='Evaluate the sensitivity analysis experiments')
        sp.setup_args(self.evaluate)
        self._modify_evaluate(sp, skip=['prepare', 'output'])
        sp.create_arguments()

        # plot parser
        sp = sps.add_parser(
            'plot', help='Plot the results sensitivity analysis experiments')
        sp.setup_args(SensitivityAnalysis.plot)
        sp.update_arg('indicators', short='ind', nargs='+',
                      metavar='indicator',
                      choices=['rsquared', 'slope', 'ks', 'quality'])
        sp.update_arg('variables', short='v', nargs='+',
                      metavar='variable',
                      choices=['prcp', 'tmin', 'tmax', 'mean_cloud'])
        sp.update_arg('meta', metavar='<yaml-file>')
        tasks = utils.unique_everseen(
            SensitivityPlot.get_manager().sort_by_requirement(
                SensitivityPlot._registry[::-1]), lambda t: t.name)
        plot_sps = sp.add_subparsers(help='Plotting tasks', chain=True)
        for task in tasks:
            plot_sp = plot_sps.add_parser(task.name, help=task.summary)
            task._modify_parser(plot_sp)
            plot_sp.create_arguments()
        sp.create_arguments()

        # remove parser
        sp = sps.add_parser('remove', help="Remove the sensitivity model")
        sp.setup_args(SensitivityAnalysis.remove)
        sp.setup_args(self.remove)
        self._modify_remove(sp)
        sp.pop_arg('modelname')
        sp.pop_arg('complete')
        sp.create_arguments()

    # -------------------------------------------------------------------------
    # -------------------------- Paths management -----------------------------
    # -------- Helper methods to cope relative and absolute paths -------------
    # -------------------------------------------------------------------------

    @docstrings.get_sectionsf('ModelOrganizer.fix_paths')
    @docstrings.dedent
    def fix_paths(self, d, root=None, model=None):
        """
        Fix the paths in the given dictionary to get absolute paths

        Parameters
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

        Parameters
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

    def _get_all_paths(self, d, base=''):
        ret = OrderedDict()
        paths = self.paths
        if base:
            base += '.'
        for key, val in d.items():
            if isinstance(val, dict):
                for key2, val2 in self._get_all_paths(val, str(key)).items():
                    ret[base + key2] = val2
            elif key in paths:
                ret[base + key] = safe_list(val)
        return ret

    def abspath(self, path, model=None, root=None):
        """Returns the path from the current working directory

        We only store the paths relative to the root directory of the model.
        This method fixes those paths to be usable from the working
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

    # -------------------------------------------------------------------------
    # -------------------------- Parser management ----------------------------
    # -------- Methods to organizer the parsing from command line -------------
    # -------------------------------------------------------------------------

    def setup_parser(self, parser=None, subparsers=None):
        """
        Create the argument parser for this instance

        This method uses the functions defined in the :attr:`commands`
        attribute to create a command line utility via the
        :class:`FuncArgParser` class. Each command in the :attr:`commands`
        attribute is interpreted as on subparser and setup initially via the
        :meth:`FuncArgParser.setup_args` method. You can modify the parser
        for each command *cmd* by including a ``_modify_cmd`` method that
        accepts the subparser as an argument

        Parameters
        ----------
        parser: FuncArgParser
            The parser to use. If None, a new one will be created
        subparsers: argparse._SubParsersAction
            The subparsers to use. If None, the
            :attr:`~ArgumentParser.add_subparser` method from `parser` will be
            called

        Returns
        -------
        FuncArgParser
            The created command line parser or the given `parser`
        argparse._SubParsersAction
            The created subparsers action or the given `subparsers`
        dict
            A mapping from command name in the :attr:`commands` attribute to
            the corresponding command line parser

        See Also
        --------
        parse_args"""
        commands = self.commands[:]
        parser_cmds = self.parser_commands.copy()

        if subparsers is None:
            if parser is None:
                parser = FuncArgParser(self.name,
                                       formatter_class=RawTextHelpFormatter)
            subparsers = parser.add_subparsers(chain=True)

        ret = {}
        for i, cmd in enumerate(commands[:]):
            func = getattr(self, cmd)
            parser_cmd = parser_cmds.setdefault(cmd, cmd.replace('_', '-'))
            ret[cmd] = sp = subparsers.add_parser(
                parser_cmd, formatter_class=RawTextHelpFormatter,
                help=docstrings.get_summary(func.__doc__ or ''))
            sp.setup_args(func)
            modifier = getattr(self, '_modify_' + cmd, None)
            if modifier is not None:
                modifier(sp)
        self.parser_commands = parser_cmds
        parser.setup_args(self.main)
        self._modify_main(parser)
        self.parser = parser
        self.subparsers = ret
        return parser, subparsers, ret

    def _finish_parser(self):
        """Create the arguments of the :attr:`parser` attribute"""
        # create the arguments
        self.parser.create_arguments()
        for parser in self.subparsers.values():
            parser.create_arguments()
        self._parser_set_up = True

    @docstrings.dedent
    def parse_args(self, args=None):
        """
        Parse the arguments from the command line (or directly) to the parser
        of this organizer

        Parameters
        ----------
        args: list
            A list of arguments to parse. If None, the :attr:`sys.argv`
            argument is used

        Returns
        -------
        %(ModelOrganizer.start.returns)s
        """
        if self.parser is None:
            self.setup_parser()
        if not self._parser_set_up:
            self._finish_parser()
        for d in chain(self.config.experiments.values(),
                       self.config.models.values()):
            if not utils.isstring(d):
                self.fix_paths(d)
        ret = self.start(**vars(self.parser.parse_args(args)))
        for d in chain(self.config.experiments.values(),
                       self.config.models.values()):
            if not utils.isstring(d):
                self.rel_paths(d)
        return ret

    # -------------------------------------------------------------------------
    # ------------------------------ Miscallaneous ----------------------------
    # -------------------------------------------------------------------------

    def _get_stations(self, stations, other_exp=False, odir=None,
                      config_key=None):
        """
        Get the stations for the parameterization or evaluation

        Parameters
        ----------
        stations: str or list of str
            either a list of stations to use or a filename containing a
            1-row table with stations
        other_exp: str
            Use the configuration from another experiment
        odir: str
            The output directory in case a list of stations is provided
        config_key:
            The key in the :attr:`exp_config` configuration dictionary holding
            information on the stations
        """
        import numpy as np

        exp_dict = self.exp_config
        fname = osp.join(odir, 'stations.dat') if odir else ''
        if other_exp and stations is None:
            stations = self.fix_paths(
                self.config.experiments[other_exp]).get(config_key)
        if isinstance(stations, six.string_types):
            stations = [stations]
        if stations is None:
            try:
                fname = exp_dict[config_key]
            except KeyError:
                raise ValueError('No stations file specified!')
            else:
                stations = np.loadtxt(exp_dict[config_key],
                                      dtype='S300', usecols=[0]).astype(
                    np.str_)
        elif len(stations) == 1 and osp.exists(stations[0]):
            fname_use = stations[0]
            exists = osp.exists(fname) if fname else False
            if exists and not osp.samefile(fname, fname_use):
                os.remove(fname)
                self._link(fname_use, fname)
            elif not exists and fname:
                self._link(fname_use, fname)
            stations = np.loadtxt(
                fname_use, dtype='S300', usecols=[0]).astype(np.str_)
        elif len(stations) and fname:
            np.savetxt(fname, stations, fmt='%s')
        if config_key and (not exp_dict.get(config_key) or not osp.samefile(
                fname, exp_dict[config_key])):
            exp_dict[config_key] = fname
        return stations

    def _setup_manager(
            self, manager, stations=None, other_exp=None,
            setup_from=None, to_db=None, to_csv=None, database=None,
            to_return=None, complete=False, base_kws={}):
        """
        Setup the data in a task manager

        This method is called by :meth:`param` and :meth:`evaluate` to setup
        the data in the given `manager`

        Parameters
        ----------
        manager: gwgen.utils.TaskManager
            The manager of the tasks to set up
        stations: list of str
            a list of stations to use
        other_exp: str
            Use the configuration from another experiment instead of
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
        database: str
            The name of a postgres data base to write the data to
        to_return: list of str
            The names of the tasks to return. If None, only the ones with an
            :attr:`gwgen.utils.TaskBase.has_run` are returned.
        complete: bool
            If True, setup and run all possible tasks
        base_kws: dict
            The dictionary with mapping from each task name to the
            corresponding initialization keywords
        """
        if complete:
            for task in manager.base_task._registry:
                base_kws.setdefault(task.name, {})
        experiment = self.experiment
        exp_dict = self.fix_paths(self.config.experiments[experiment])
        if database is not None:
            exp_dict['database'] = database
        # setup up the keyword arguments for the parameterization tasks
        for key, d in base_kws.items():
            if d.get('setup_from') is None:
                d['setup_from'] = setup_from
            if to_csv:
                d['to_csv'] = to_csv
            elif to_csv is None and d.get('to_csv') is None:
                # delete the argument if the subparser doesn't use it
                d.pop('to_csv', None)
            if to_db:
                # delete the argument if the subparser doesn't use it
                d['to_db'] = to_db
            elif to_db is None and d.get('to_db') is None:
                d.pop('to_db', None)
            if other_exp and not d.get('other_exp'):
                d['other_exp'] = other_exp
            exp = d.pop('other_exp', experiment) or experiment
            d['config'] = self.fix_paths(self.config.experiments[exp])
            d['model_config'] = self.config.models[d['config']['model']]
            for key in ['stations', 'complete', 'norun', 'other_id',
                        'database']:
                d.pop(key, None)
            self._update_model_with_globals(self.fix_paths(d['model_config']))
        # choose keywords for data processing
        manager.initialize_tasks(stations, task_kws=base_kws)
        manager.setup(stations, to_return=to_return)

    def _modify_task_parser(self, parser, base_task, skip=None, only=None):
        def norun(s):
            if s is True or s == 'all':
                return True
            try:
                return bool(int(s))
            except TypeError:
                return s.split(',')
        skip = skip or []
        if only is None:
            def key_func(t):
                return t.name not in skip
        else:
            def key_func(t):
                return t.name in only and t.name not in skip
        self._modify_main(parser)
        parser.update_arg('setup_from', short='f', long='from',
                          dest='setup_from')
        parser.update_arg('other_exp', short='ido', long='other_id',
                          dest='other_exp')
        try:
            parser.update_arg('stations', short='s')
        except KeyError:
            pass
        parser.update_arg('database', short='db')
        parser.pop_arg('to_return', None)
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

        tasks = filter(key_func, utils.unique_everseen(
            base_task.get_manager().sort_by_requirement(
                base_task._registry[::-1]), lambda t: t.name))
        sps = parser.add_subparsers(title='Tasks', chain=True)
        for task in tasks:
            sp = sps.add_parser(task.name, help=task.summary,
                                formatter_class=RawTextHelpFormatter)
            task._modify_parser(sp)
            sp.add_argument(
                '-ido', '--other_id', help=other_exp_doc,
                metavar=other_exp_dtype)
            sp.create_arguments()

    def _update_model_with_globals(self, d):
        datadir = self.config.global_config.get('data')
        if datadir and 'data' not in d:
            d['data'] = datadir
        return d

    def is_archived(self, experiment, ignore_missing=True):
        """
        Convenience function to determine whether the given experiment has been
        archived already

        Parameters
        ----------
        experiment: str
            The experiment to check

        Returns
        -------
        str or None
            The path to the archive if it has been archived, otherwise None
        """
        if ignore_missing:
            if utils.isstring(self.config.experiments.get(experiment, True)):
                return self.config.experiments.get(experiment, True)
        else:
            if utils.isstring(self.config.experiments[experiment]):
                return self.config.experiments[experiment]

    @staticmethod
    def _archive_extensions():
        """Create translations from file extension to archive format

        Returns
        -------
        dict
            The mapping from file extension to archive format
        dict
            The mapping from archive format to default file extension
        """
        ext_map = {}
        fmt_map = {}
        for key, exts, desc in shutil.get_unpack_formats():
            fmt_map[key] = exts[0]
            for ext in exts:
                ext_map[ext] = key
        return ext_map, fmt_map

    def _link(self, source, target):
        """Link two files

        Parameters
        ----------
        source: str
            The path of the source file
        target: str
            The path of the target file"""
        if self.global_config.get('copy', True) and osp.isfile(source):
            shutil.copyfile(source, target)
        elif self.global_config.get('use_relative_links', True):
            os.symlink(osp.relpath(source, osp.dirname(target)), target)
        else:
            os.symlink(osp.abspath(source), target)

    def __reduce__(self):
        return self.__class__, (self.name, self.config), {
            '_experiment': self._experiment, '_modelname': self._modelname,
            'no_modification': self.no_modification}


def _get_parser():
    """Function returning the gwgen parser, necessary for sphinx documentation
    """
    organizer = ModelOrganizer('gwgen')
    organizer.setup_parser()
    organizer._finish_parser()
    return organizer.parser


def main(args=None):
    organizer = ModelOrganizer('gwgen')
    organizer.parse_args(args)
    if not organizer.no_modification:
        organizer.config.save()


if __name__ == '__main__':
    main()
