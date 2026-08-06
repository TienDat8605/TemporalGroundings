from hybrid_vtg.config import CoarseConfig, PipelineConfig, RefinementConfig, SemVIDConfig
from hybrid_vtg.pipeline import HybridVTGPipeline, _SampleContext
from hybrid_vtg.types import Component, GroundingPrediction, Sample, TemporalRoute


def _context(sample_id: str, durations: tuple[float, ...]) -> _SampleContext:
    sample = Sample(sample_id, f"{sample_id}.mp4", f"/{sample_id}.mp4", 60.0, "query")
    components = tuple(Component(index * 10.0, index * 10.0 + duration, 1.0) for index, duration in enumerate(durations))
    return _SampleContext(
        sample=sample, started=0.0,
        temporal_route=TemporalRoute(components, (), 0.0, False, sum(durations)),
        ordered_components=components,
    )


def test_microbatch_scheduler_pairs_similar_loads_and_keeps_odd_tail():
    pipeline = HybridVTGPipeline.__new__(HybridVTGPipeline)
    pipeline.config = PipelineConfig(semvid=SemVIDConfig(batch_size=2))
    batches = pipeline._component_batches([_context("a", (12.0, 2.0)), _context("b", (3.0,))])
    assert [[task.request.component.duration for task in batch] for batch in batches] == [[2.0, 3.0], [12.0]]


def test_prefetched_pipeline_restores_dataset_order():
    class FakeGrounder:
        @staticmethod
        def prepare_batch(requests, **_kwargs):
            return tuple(requests)

        @staticmethod
        def ground_prepared(requests):
            return [
                GroundingPrediction(
                    (request.component.start + 1.0, request.component.end - 1.0),
                    request.component, "{}", {}, {},
                    {"component_seconds": 0.1, "qwen_batch_size": len(requests)},
                )
                for request in requests
            ]

    pipeline = HybridVTGPipeline.__new__(HybridVTGPipeline)
    pipeline.config = PipelineConfig(
        coarse=CoarseConfig(enabled=False),
        semvid=SemVIDConfig(batch_size=2, preprocess_workers=1, prefetch_depth=2),
        refinement=RefinementConfig(enabled=False),
    )
    pipeline.grounder = FakeGrounder()
    pipeline.coarse_encoder = None
    samples = [
        Sample(str(index), f"{index}.mp4", f"/{index}.mp4", 10.0, "query")
        for index in range(3)
    ]
    outcomes = list(pipeline.iter_results(samples))
    assert [sample.id for sample, _, _ in outcomes] == ["0", "1", "2"]
    assert all(error is None and record is not None for _, record, error in outcomes)
