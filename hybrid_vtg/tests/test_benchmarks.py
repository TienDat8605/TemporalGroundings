import json

from hybrid_vtg.benchmarks import load_tacos


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
