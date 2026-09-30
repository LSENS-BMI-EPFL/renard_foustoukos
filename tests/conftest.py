"""Test configuration: point fast_learning at a throw-away config before any
module reads it, so the tests need no data and no config.yaml."""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix='fast_learning_tests_')
_config = os.path.join(_tmp, 'config.yaml')
with open(_config, 'w') as f:
    f.write(f'data_root: {_tmp}/data\noutput_root: {_tmp}/outputs\n')
os.environ['FAST_LEARNING_CONFIG'] = _config
os.environ.setdefault('MPLBACKEND', 'Agg')
