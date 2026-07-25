import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import compare_omtg_inference
import run_omtg_2fps_baseline
import run_omtg_search
from vlmeval.dataset.omtgbench import compute_one_to_many_metrics, parse_time_intervals
from vlmeval.omtg_search import (
    consolidate_intervals,
    distribute_frames,
    estimated_sampled_frames,
    grounding_prompt,
    query_text,
    residual_frame_policy,
    residual_priorities,
    retained_window_count,
    router_recall,
    read_jsonl,
    uniform_windows,
    windows_from_boundaries,
)
from vlmeval.vlm.qwen3_vl_embedding import Qwen3VLForEmbedding
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig


class OMTGMetricsTest(unittest.TestCase):
    def test_embedding_model_has_runtime_config_class(self):
        self.assertIs(Qwen3VLForEmbedding.config_class, Qwen3VLConfig)

    def test_multispan_exact_match(self):
        segments = [[1.0, 2.0], [4.5, 7.0]]
        metrics = compute_one_to_many_metrics(segments, segments)
        self.assertEqual(metrics['C-Acc'], 1.0)
        self.assertEqual(metrics['EtF1'], 1.0)
        self.assertEqual(metrics['tIoU'], 1.0)

    def test_parser_accepts_json_array(self):
        self.assertEqual(
            parse_time_intervals('[[1, 2], [4.5, 7]]'),
            [[1.0, 2.0], [4.5, 7.0]],
        )

    def test_parser_accepts_timelens_start_end_format(self):
        self.assertEqual(
            parse_time_intervals('Start: 9.0, End: 49.0'),
            [[9.0, 49.0]],
        )

    def test_wrong_cardinality_zeroes_etf1(self):
        metrics = compute_one_to_many_metrics([[1, 2]], [[1, 2], [4, 5]])
        self.assertEqual(metrics['C-Acc'], 0.0)
        self.assertEqual(metrics['EtF1'], 0.0)
        self.assertEqual(metrics['CardinalityError'], 1.0)


class SearchPolicyTest(unittest.TestCase):
    def test_shared_controlled_prompt(self):
        question = (
            "Find the video segment that corresponds to the given textual query "
            "'a person waves' and determine its start and end seconds."
        )
        self.assertEqual(query_text(question), 'a person waves')
        expected = grounding_prompt('a person waves', 12.5, local=False)
        self.assertEqual(
            run_omtg_2fps_baseline.prompt_for_row({'question': question}, 'controlled', 12.5),
            expected,
        )
        self.assertEqual(
            run_omtg_2fps_baseline.prompt_for_row({'question': question}, 'official', 12.5),
            question,
        )

    def test_estimated_frames_are_factor_aligned(self):
        self.assertEqual(estimated_sampled_frames(10.4, 2.0), 20)
        self.assertEqual(estimated_sampled_frames(0.1, 2.0), 2)

    def test_content_boundaries_obey_window_limits(self):
        windows = windows_from_boundaries(150, [12, 30, 47, 89, 120])
        self.assertEqual(windows[0].start, 0.0)
        self.assertEqual(windows[-1].end, 150.0)
        self.assertTrue(all(20 <= window.duration <= 60 for window in windows))

    def test_uniform_fallback_covers_video(self):
        windows = uniform_windows(150)
        self.assertEqual(windows[0].start, 0.0)
        self.assertEqual(windows[-1].end, 150)
        self.assertTrue(all(window.duration >= 20 for window in windows))

    def test_k_rule_and_allocation(self):
        self.assertEqual(retained_window_count(1), 1)
        self.assertEqual(retained_window_count(9), 3)
        self.assertEqual(retained_window_count(100), 8)
        self.assertEqual(sum(distribute_frames(32, 3)), 32)

    def test_merge_and_router_recall(self):
        merged = consolidate_intervals([[1, 3], [1.1, 3.1], [4, 5]], duration=10)
        self.assertEqual(merged, [[1.0, 5.0]])
        windows = uniform_windows(150)
        self.assertEqual(router_recall(windows, [0], [[1, 2], [100, 101]]), 0.5)

    def test_resume_repairs_partial_final_jsonl_record(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            path.write_text('{"id": 1}\n{"id":', encoding='utf-8')
            self.assertEqual(read_jsonl(path, ('id',)), {(1,): {'id': 1}})
            self.assertEqual(path.read_text(encoding='utf-8'), '{"id": 1}\n')


class ScheduleIntegrationTest(unittest.TestCase):
    class FakeGrounder:
        def generate(self, message, dataset=None):
            return '[[1, 2], [4, 5]]'

    def test_all_schedules_execute_and_account_frames(self):
        route = {
            'duration': 150.0,
            'windows': [
                {'start': 0.0, 'end': 50.0},
                {'start': 48.0, 'end': 100.0},
                {'start': 98.0, 'end': 150.0},
            ],
            'k': 2,
            'selected': [0, 2],
            'scores': [0.9, 0.4, 0.7],
            'embedding_frames': 12,
            'telemetry': {'gpu_seconds': 0.5, 'wall_seconds': 0.6, 'peak_vram_bytes': 100},
        }
        row = {'id': 0, 'video': 'fake.mp4', 'question': 'find it', 'answer': '[[1,2]]'}

        def fake_extract(video_path, timestamps, destination, max_width=336):
            return [f'/tmp/frame_{index}.jpg' for index in range(len(timestamps))]

        with patch.object(run_omtg_search, 'extract_frames', side_effect=fake_extract):
            for schedule in run_omtg_search.ALL_SCHEDULES:
                result = run_omtg_search.execute_schedule(
                    model=self.FakeGrounder(), schedule=schedule, budget=32, row=row, route=route,
                    video_path=Path('/tmp/fake.mp4'), cache_root=Path('/tmp/cache'), frame_width=336,
                )
                self.assertTrue(result['prediction'])
                if schedule in run_omtg_search.RESIDUAL_SCHEDULES:
                    self.assertLessEqual(result['total_frames'], 32)
                else:
                    self.assertEqual(result['total_frames'], 32)
                self.assertEqual(result['budget_overflow_frames'], 0)

    def test_residual_schedules_never_exceed_observed_budgets(self):
        for budget in (64, 128):
            for count in range(1, 22):
                policy = residual_frame_policy(budget, count)
                used = policy['router_frames'] + (
                    policy['maximum_actions'] * policy['local_frames_per_action']
                )
                self.assertLessEqual(used, budget)

    def test_residual_priority_is_deterministic_and_label_free(self):
        windows = [run_omtg_search.Window(0, 10), run_omtg_search.Window(10, 20)]
        first = residual_priorities(windows, [0.2, 0.8], [], [])
        self.assertGreater(first[1]['utility'], first[0]['utility'])
        second = residual_priorities(windows, [0.2, 0.8], [1], [[10, 20]])
        self.assertEqual(list(second), [0])


class ComparisonReportTest(unittest.TestCase):
    def test_comparison_separates_accuracy_and_compute_claims(self):
        metrics = {
            'C-Acc': 10.0,
            'EtF1': 8.0,
            'tF1@0.3': 30.0,
            'tF1@0.5': 20.0,
            'tF1@0.7': 10.0,
            'tIoU': 40.0,
            'CardinalityError': 2.0,
        }
        baseline = {
            'model': 'MCG-NJU/TimeLens2-4B',
            'results': [
                {'schedule': 'paper-2fps-official', 'complete': True, **metrics},
                {
                    'schedule': 'paper-2fps-controlled',
                    'complete': True,
                    **metrics,
                    'EtF1': 9.0,
                },
            ],
        }
        search = {
            'model': 'MCG-NJU/TimeLens2-4B',
            'results': [
                {
                    'schedule': 'uniform-one-shot',
                    'budget': 64,
                    'complete': True,
                    'gpu_seconds': 100,
                    **metrics,
                },
                {
                    'schedule': 'embedding-window-local',
                    'budget': 64,
                    'complete': True,
                    'gpu_seconds': 110,
                    **metrics,
                    'EtF1': 18.0,
                },
            ],
        }
        result = compare_omtg_inference.build_comparison(baseline, search)
        comparisons = {item['name']: item for item in result['comparisons']}
        self.assertEqual(
            comparisons['hierarchical-vs-uniform-at-64-frames']['comparison_class'],
            'pareto',
        )
        self.assertEqual(
            comparisons['embedding-window-local-64f-vs-paper-2fps-controlled'][
                'comparison_class'
            ],
            'accuracy-only-different-input-policy',
        )
        self.assertEqual(
            comparisons['controlled-prompt-vs-official-prompt-at-2fps']['delta_EtF1'],
            1.0,
        )


if __name__ == '__main__':
    unittest.main()
