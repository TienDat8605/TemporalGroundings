from hybrid_vtg.config import GrounderConfig, PipelineConfig
from hybrid_vtg.pipeline import HybridVTGPipeline
from hybrid_vtg.types import GroundingPrediction, Sample


def _sample(sample_id: str, duration: float) -> Sample:
    return Sample(sample_id, f"{sample_id}.mp4", f"/{sample_id}.mp4", duration, "query")


def test_microbatch_scheduler_pairs_similar_video_lengths_and_keeps_tail():
    pipeline = HybridVTGPipeline.__new__(HybridVTGPipeline)
    pipeline.config = PipelineConfig(grounder=GrounderConfig(batch_size=2))
    batches = pipeline._batches([_sample("long", 12.0), _sample("short", 2.0), _sample("mid", 3.0)])
    assert [[task.sample.duration for task in batch] for batch in batches] == [[2.0, 3.0], [12.0]]
    assert all(task.request.component.start == 0.0 for batch in batches for task in batch)


def test_prefetched_pipeline_restores_dataset_order():
    class FakeGrounder:
        @staticmethod
        def prepare_batch(requests, **_kwargs):
            return tuple(requests)

        @staticmethod
        def ground_prepared(requests):
            return [
                GroundingPrediction(
                    (1.0, request.component.end - 1.0), request.component, "{}", {}, {}, {},
                    intervals=((1.0, request.component.end - 1.0),),
                )
                for request in requests
            ]

    pipeline = HybridVTGPipeline.__new__(HybridVTGPipeline)
    pipeline.config = PipelineConfig(grounder=GrounderConfig(
        batch_size=2, preprocess_workers=1, prefetch_depth=2,
    ))
    pipeline.grounder = FakeGrounder()
    samples = [_sample("0", 10.0), _sample("1", 8.0), _sample("2", 9.0)]
    outcomes = list(pipeline.iter_results(samples))
    assert [sample.id for sample, _, _ in outcomes] == ["0", "1", "2"]
    assert all(error is None and record is not None for _, record, error in outcomes)
    assert all(record["efficiency"]["qwen_batch_size"] == 1 for _, record, _ in outcomes)
