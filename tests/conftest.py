
def pytest_addoption(parser):
    group = parser.getgroup("gwgen", "GWGEN specific options")
    group.addoption('--offline', help="Block outgoing internet connections",
                    action='store_true')
    group.addoption('--no-db', help="Don't test postgres databases",
                    action='store_true')


def pytest_configure(config):
    if config.getoption('offline'):
        import _base_testing as bt
        bt.online = False
    if config.getoption('no_db'):
        import _base_testing as bt
        bt.use_db = False
