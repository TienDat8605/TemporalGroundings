import json

from hybrid_vtg.benchmarks import load_omtg, load_tacos


def test_load_tacos_jsonl(tmp_path):
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "s30-d52.avi"
    video.touch()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    record = {
        "qid": "s30-d52_0",
        "vid": "s30-d52",
        "query": "She took out kiwi",
        "duration": 249.8,
        "relevant_windows": [[4.8, 12.0]],
    }
    (annotations / "test.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    samples = load_tacos(tmp_path, "test")

    assert len(samples) == 1
    assert samples[0].id == "s30-d52_0"
    assert samples[0].targets == ((4.8, 12.0),)
    assert samples[0].video_path == str(video)


def test_load_omtg_tsv_as_native_multispan(monkeypatch, tmp_path):
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "clip.mp4"
    video.touch()
    (tmp_path / "OMTGBench.tsv").write_text(
        "id\tvideo\tquestion\tanswer\n"
        "0\tclip.mp4\tFind the video segment for the given textual query 'a person's hand waves' "
        "and determine its start and end seconds.\t"
        '[[1.0, 2.0], [4.0, 5.0]]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hybrid_vtg.benchmarks.probe_video",
        lambda _path: type("Metadata", (), {"duration": 10.0})(),
    )
    samples = load_omtg(tmp_path)
    assert samples[0].query == "a person's hand waves"
    assert samples[0].targets == ((1.0, 2.0), (4.0, 5.0))
    assert samples[0].cardinality == "multi"
    assert samples[0].duration == 10.0
