import json

from vlmeval import benchmark_adapters as adapters


def test_charades_adapter_emits_canonical_rows(tmp_path, monkeypatch):
    (tmp_path / 'videos').mkdir()
    (tmp_path / 'videos' / 'VID1.mp4').touch()
    (tmp_path / 'charades_sta_test.txt').write_text(
        'VID1 1.5 4.0##person opens a door\n', encoding='utf-8'
    )
    monkeypatch.setattr(adapters, '_duration', lambda _: 10.0)
    rows, metadata = adapters.load_adapter('charades-sta', tmp_path, split='test')
    adapters.validate_canonical_rows(rows)
    assert rows[0]['targets'] == [[1.5, 4.0]]
    assert rows[0]['cardinality'] == 'single'
    assert metadata['split'] == 'test'


def test_activitynet_adapter_preserves_split_and_events(tmp_path, monkeypatch):
    (tmp_path / 'videos').mkdir()
    (tmp_path / 'videos' / 'abc.mp4').touch()
    (tmp_path / 'val_1.json').write_text(json.dumps({
        'v_abc': {
            'duration': 12.0,
            'timestamps': [[1.0, 3.0], [5.0, 8.0]],
            'sentences': ['first event', 'second event'],
        }
    }), encoding='utf-8')
    monkeypatch.setattr(adapters, '_duration', lambda _: 12.0)
    rows, metadata = adapters.load_adapter(
        'activitynet-captions', tmp_path, split='val_1'
    )
    adapters.validate_canonical_rows(rows)
    assert len(rows) == 2
    assert {row['query'] for row in rows} == {'first event', 'second event'}
    assert metadata['split'] == 'val_1'
