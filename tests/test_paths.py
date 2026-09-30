import importlib

import pytest


def _load_paths(tmp_path, monkeypatch, text):
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(text)
    monkeypatch.setenv('FAST_LEARNING_CONFIG', str(cfg))
    import fast_learning.paths as paths

    return importlib.reload(paths)


def test_default_layout(tmp_path, monkeypatch):
    paths = _load_paths(tmp_path, monkeypatch, 'data_root: /data\noutput_root: /out\n')
    assert paths.nwb_dir == '/data/nwb'
    assert paths.tensor_dir == '/data/processed/mice'
    assert paths.db_path == '/data/metadata/sessions.csv'
    assert paths.processed_dir == '/out/processed'
    assert paths.figures_dir == paths.manuscript_output_dir == '/out/figures'
    assert paths.figure_formats == ['pdf']


def test_overrides(tmp_path, monkeypatch):
    paths = _load_paths(
        tmp_path,
        monkeypatch,
        'data_root: /data\noutput_root: /out\ntensor_dir: /elsewhere/mice\nfigure_formats: [pdf, svg]\n',
    )
    assert paths.tensor_dir == '/elsewhere/mice'
    assert paths.figure_formats == ['pdf', 'svg']


def test_missing_required_key(tmp_path, monkeypatch):
    with pytest.raises(KeyError, match='output_root'):
        _load_paths(tmp_path, monkeypatch, 'data_root: /data\n')


def teardown_module():
    # monkeypatch has restored the session test config: reload paths with it.
    import fast_learning.paths as paths

    importlib.reload(paths)
