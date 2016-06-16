from __future__ import print_function
import os
import os.path as osp
from shutil import copyfile, rmtree
import re
import sys
import datetime as dt
from itertools import groupby, chain
from argparse import ArgumentParser, Namespace
import inspect
import logging
from psyplot.compat.pycompat import OrderedDict
from psyplot.docstring import DocStringProcessor
from gwgen.config import Config, ordered_yaml_dump

docstrings = DocStringProcessor()


class FuncArgParser(ArgumentParser):
    """Subclass of an argument parser that get's parts of the information
    from a given function"""

    def __init__(self, *args, **kwargs):
        self.__subparsers = None
        super(FuncArgParser, self).__init__(*args, **kwargs)
        self.__arguments = OrderedDict()
        self.__funcs = []
        self.__main = None

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
            arg_doc = docstrings._keep_params(doc, [arg]) or \
                docstrings._keep_types(doc, [arg])
            args_dict[arg] = d = {'dest': arg, 'short': arg, 'long': arg}
            if arg_doc:
                lines = arg_doc.splitlines()
                d['help'] = '\n'.join(lines[1:])
                metavar = lines[0].split(':', 1)
                if i >= default_min:
                    d['default'] = defaults[i - default_min]
                else:
                    d['positional'] = True
                if len(metavar) > 1:
                    dtype = metavar[1].strip()
                    if dtype == 'bool' and 'default' in d:
                        d['action'] = 'store_false' if d['default'] else \
                            'store_true'
                    else:
                        d['metavar'] = metavar[1].strip()

    def update_arg(self, arg, if_existent=True, **kwargs):
        """Update the `add_argument` data for the given parameter
        """
        if not if_existent:
            self.__arguments.setdefault(arg, kwargs)
        self.__arguments[arg].update(kwargs)

    def pop_key(self, arg, key, *args, **kwargs):
        """Delete a previously defined key for the `add_argument`
        """
        return self.__arguments[arg].pop(key, *args, **kwargs)

    def create_arguments(self):
        """Create and add the arguments"""
        ret = []
        for arg, d in self.__arguments.items():
            try:
                is_positional = d.pop('positional', False)
                short = d.pop('short')
                long_name = d.pop('long', None)
                if short == long_name:
                    long_name = None
                args = [short, long_name] if long_name else [short]
                if not is_positional:
                    for i, arg in enumerate(args):
                        args[i] = '-' * (i + 1) + arg
                else:
                    d.pop('dest', None)
                group = d.pop('group', self)
                ret.append(group.add_argument(*args, **d))
            except Exception:
                print('Error while creating argument %s' % arg)
                raise
        return ret

    def add_subparsers(self, *args, **kwargs):
        ret = super(FuncArgParser, self).add_subparsers(*args, **kwargs)
        self.__subparsers = ret
        return ret

    def parse_known_args(self, args=None, namespace=None):
        if self.__subparsers is not None:
            commands = list(self.__subparsers.choices.keys())
            # get the first argument to make sure that everything works
            if args is None:
                args = sys.argv[1:]

            def groupargs(arg, currentarg=['main']):
                if arg in commands:
                    currentarg[0] = arg
                return currentarg[0]
            choices_d = OrderedDict()
            remainders = OrderedDict()
            main_args = []
            i = 0
            cmd = 'main'
            for i, (cmd, subargs) in enumerate(groupby(args, groupargs)):
                if cmd == 'main':
                    main_args += list(subargs)
                else:
                    choices_d[cmd], remainders[cmd] = super(
                        FuncArgParser, self).parse_known_args(
                            main_args + list(subargs))
            if i == 0 and cmd == 'main':  # only main is parsed
                choices_d[cmd], remainders[cmd] = super(
                            FuncArgParser, self).parse_known_args(
                                main_args)
            return Namespace(**choices_d), list(chain(*remainders.values()))
        # otherwise, use the default behaviour
        return super(FuncArgParser, self).parse_known_args(args, namespace)


class ModelOrganizer(object):
    """
    A class for organizing a model

    This class is indended to have hold the basic functions for organizing a
    model. You can subclass the functions ``setup, init`` to fit to your model.
    When using the model from the command line, you can also use the
    :meth:`setup_parser` method to create the argument parsers"""

    commands = ['setup', 'compile', 'init', 'info', 'remove', 'param']

    #: The :class:`gwgen.parser.FuncArgParser` to use for initializing the
    #: model. This attribute is set by the :meth:`setup_parser` method and used
    #: by the `start` method
    parser = None

    #: list of str. The keys describing paths for the model
    paths = ['expdir']

    _modelname = None
    _experiment = None

    @property
    def modelname(self):
        """The name of the model that is currently processed"""
        if self._modelname is None:
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
    def logger(self):
        """The logger of this organizer"""
        return logging.getLogger(
            '.'.join([__name__, self.name, self._modelname or '',
                      self._experiment or '']))

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
    def main(self, modelname=None, experiment=None, last=False, new=False,
             verbose=None):
        """
        The main function for parsing global arguments

        Parameters
        ----------
        modelname: str
            The name of the model that shall be used. If None, it is determined
            by the experiment or the ``setup`` sub command
        experiment: str
            The id of the experiment to use
        last: bool
            If True, the last experiment is used
        new: bool
            If True, a new experiment is created
        verbose: str or int
            The verbosity level to use. Either one of ``'DEBUG', 'INFO',
            'WARNING', 'ERROR'`` or the corresponding integer (see the pythons
            logging module)"""
        self.modelname = modelname or self.modelname
        if last and self.config.experiments:
            self.experiment = None
        elif new and self.config.experiments:
            self.experiment = self._get_next_name(self.experiment)
        elif experiment:
            self.experiment = experiment
        if verbose is not None:
            if verbose in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                verbose = getattr(logging, verbose)
            logging.getLogger(__name__).setLevel(verbose)

    def _modify_main(self, parser):
        to_update = {
            'modelname': dict(short='m'),
            'experiment': dict(short='id'),
            'last': dict(short='l'),
            'new': dict(short='n'),
            'verbose': dict(short='v', const='DEBUG', nargs='?', default=None),
            }
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
        self.main(modelname=modelname, **kwargs)
        root_dir = osp.abspath(osp.join(root_dir, modelname))
        models[modelname] = OrderedDict([
            ('root', root_dir), ('timestamps', OrderedDict())])
        src_dir = osp.join(root_dir, 'src')
        models[modelname]['src'] = self.relpath(src_dir)
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
        mdict['bin'] = bin_dir = osp.join(mdict['root'], 'bin')
        src_dir = self.abspath(mdict['src'])
        if not os.path.exists(bin_dir):
            self.logger.debug("    Creating bin directory %s", bin_dir)
            os.makedirs(bin_dir)
        self.logger.debug("    Linking files...")
        for f in os.listdir(src_dir):
            target = osp.join(bin_dir, f)
            if osp.exists(target):
                os.remove(target)
            os.symlink(osp.relpath(osp.join(src_dir, f), bin_dir), target)
        spr.call(['make', '-C', bin_dir, 'all'])

    docstrings.keep_params('ModelOrganizer.main.parameters', 'experiment')

    @docstrings.dedent
    def init(self, experiment=None, description=None, **kwargs):
        """
        Initialize a new experiment

        Parameters
        ----------
        %(ModelOrganizer.main.parameters.experiment)s If None, a new
            experiment will be created.
        description: str
            A short summary of the experiment
        ``**kwargs``
            Keyword arguments passed to the :meth:`main` method
        """
        experiments = self.config.experiments
        if experiment is None and not experiments:
            experiment = self.name + '_exp0'
        elif experiment is None:
            experiment = self._get_next_name(self.experiment)
        self.main(experiment=experiment, **kwargs)
        modelname = self.modelname
        exp_dict = experiments.setdefault(experiment, OrderedDict())
        if description is not None:
            exp_dict['description'] = description
        exp_dict['model'] = modelname
        exp_dir = osp.join(
            self.config.models[modelname]['root'], 'experiments', experiment)
        exp_dict['expdir'] = self.relpath(exp_dir)
        exp_dict['timestamps'] = OrderedDict()

        if not os.path.exists(exp_dir):
            self.logger.info("Initializing experiment %s of model %s",
                             experiment, modelname)
            self.logger.debug("    Creating experiment directory %s", exp_dir)
            os.makedirs(exp_dir)
        return exp_dict

    def _modify_init(self, parser):
        self._modify_main(parser)
        parser.update_arg('description', short='d')

    @docstrings.dedent
    def info(self, complete=False, no_fix=False, **kwargs):
        """
        Print information on the experiments

        Parameters
        ----------
        complete: bool
            If True/set, the information on all experiments are printed
        no_fix: bool
            If set, paths are given relative to the root directory of the
            model
        """
        exps = self.config.experiments
        self.main(**kwargs)
        experiment = self.experiment
        if not complete:
            exps = exps[experiment]
            if not no_fix:
                for key in self.paths:
                    if key in exps:
                        exps[key] = self.abspath(exps[key])
        elif not no_fix:
            for exp, d in exps.items():
                for key in self.paths:
                    if key in d:
                        d[key] = self.abspath(d[key], d['model'])
        print(ordered_yaml_dump(exps, default_flow_style=False))
        sys.exit(0)

    def _modify_info(self, parser):
        self._modify_main(parser)
        parser.update_arg('no_fix', short='nf')
        parser.update_arg('complete', short='a', long='all', dest='complete')

    docstrings.keep_params('ModelOrganizer.main.parameters', 'experiment',
                           'modelname')

    @docstrings.dedent
    def remove(self, experiment=None, modelname=None, complete=False,
               **kwargs):
        """
        Delete an existing experiment and/or modelname

        Parameters
        ----------
        %(ModelOrganizer.main.parameters.experiment|modelname)s
        complete: bool
            If set, all experiments, models and config files are deleted

        Warnings
        --------
        This will remove the entire folder and all the related informations in
        the configurations!
        """
        self.main(experiment=experiment, modelname=modelname, **kwargs)
        all_experiments = self.config.experiments
        all_models = self.config.models
        if complete:
            experiments = list(all_experiments.keys())
        elif modelname is not None:
            experiments = [exp for exp, val in all_experiments.items()
                           if val['model'] == modelname]
        else:
            experiments = [self.experiment]
        models = set()
        for exp in experiments:
            self.logger.debug("Removing experiment %s", exp)
            exp_dict = all_experiments.pop(exp)
            if osp.exists(exp_dict['expdir']):
                rmtree(exp_dict['expdir'])
            models.add(exp_dict['model'])

        if complete or modelname is not None:
            for model in models if not complete else list(all_models.keys()):
                self.logger.debug("Removing model %s", model)
                modeldir = all_models.pop(model)['root']
                if osp.exists(modeldir):
                    rmtree(modeldir)

    def _modify_remove(self, parser):
        self._modify_main(parser)
        parser.update_arg('complete', short='a', long='all', dest='complete')

    @docstrings.get_sectionsf('ModelOrganizer._get_postgres_engine')
    @docstrings.dedent
    def _get_postgres_engine(self, database, user=None, host='127.0.0.1',
                             port=None):
        """
        Get the engine to access the given `database`

        This method creates an engine using sqlalchemy's create_engine function
        to access the given `database` via postgresql. If the database is not
        existent, it will be created

        Parameters
        ----------
        database: str
            The name of a psql database. If provided, the processed data will
            be stored
        user: str
            The username to use when logging into the database
        host: str
            the host which runs the database server
        port: int
            The port to use to log into the the database

        Returns
        -------
        sqlalchemy.engine.base.Engine
            Tha engine to access the database"""
        import sqlalchemy
        logger = self.logger
        base_str = 'postgresql://'
        if user:
            base_str += user + '@'
        base_str += host
        if port:
            base_str += ':' + port
        engine_str = base_str + '/' + database  # to create the database
        logger.debug("Creating engine with %s")
        engine = sqlalchemy.create_engine(engine_str)
        logger.debug("Try to connect...")
        try:
            conn = engine.connect()
        except sqlalchemy.exc.OperationalError:
            # data base does not exist, so create it
            logger.debug("Failed...", exc_info=True)
            logger.debug("Creating database by logging into postgres")
            pengine = sqlalchemy.create_engine(base_str + '/postgres')
            conn = pengine.connect()
            conn.execute('commit')
            conn.execute('CREATE DATABASE ' + database)
            conn.close()
        else:
            conn.close()
        return engine

    @docstrings.dedent
    def param(self, other_exp=None, complete=False,
              markov=False, precip=False, temp=False, cloud=False,
              cov=False, stations=None, datapath=None, database=None,
              user=None, host='127.0.0.1', port=None, **kwargs):
        """
        Parameterize the model

        Parameters
        ----------
        %(ModelOrganizer.main.parameters.experiment)s. If None, the last
        created experiment is used
        other_exp: str
            Use the parameterization from another experiment instead of
            calculating it
        complete: bool
            If set, all of the below parameterizations are calculated
        markov: bool
            If set, transition probabilities for the markov chain are
            calculated
        precip: bool
            If set, the hybrid Gamma-GP parameterization is calculated
        temp: bool
            If set, the wet/dry/all temperature correlations are calculated
        cloud: bool
            If set, the wet/dry/all cloud correlations are calculated
        cov: bool
            If set, the covariances between temperature and cloud are
            calculated
        stations: str or list of str
            either a list of stations to use for the parameterization or a
            filename consiting of 1 table with stations
        datapath: str
            The path to the downloaded GHCN raw data. If not given, it will be
            assumed to be in the <modeldir>/data/ghcn_all. If the data is not
            found, it will be taken from the web page
            ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz
        %(ModelOrganizer._get_postgres_engine.parameters)s
        """
        print(kwargs, cloud)
        experiment = self.experiment
        exp_dict = self.config.experiments[experiment]
        modelname = self.modelname = exp_dict['model']
        logger = self.logger
        logger.info("Parameterizing experiment %s of model %s",
                    experiment, modelname)
        if complete:
            markov = True
            precip = True
            temperature = True
            cloud = True
            cov = True
        # first we check whether everything works with the database
        if database:
            engine = self._get_engine(database, user, host, port)

    def _modify_param(self, parser):
        self._modify_main(parser)
        subparsers = parser.add_subparsers()
        cloud_parser = subparsers.add_parser('cloud', help='cloud')
        cloud_parser.add_argument('-to_db', help="Write to db")
        temp_parser = subparsers.add_parser('temp', help='cloud')
        temp_parser.add_argument('-to_db', help="Write to db")
        temp_parser.add_argument('-to_f', help="Write to file")

    def abspath(self, path, model=None):
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

        Returns
        -------
        str
            The path as it is accessible from the current working directory"""
        return osp.join(self.config.models[model or self.modelname]['root'],
                        path)

    def relpath(self, path, model=None):
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

        Returns
        -------
        str
            The path relative from the root directory"""
        return osp.relpath(
            path, self.config.models[model or self.modelname]['root'])

    def setup_parser(self, parser=None, subparsers=None):
        commands = self.commands

        if subparsers is None:
            if parser is None:
                parser = FuncArgParser(self.name)
            subparsers = parser.add_subparsers()

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
            ts.update(self.config.models[modelname]['timestamps'])
            self.config.experiments[exp]['timestamps'].update(ts)
        self.config.save()


def main():
    organizer = ModelOrganizer('gwgen')
    organizer.setup_parser()
    organizer.start()


if __name__ == '__main__':
    main()
