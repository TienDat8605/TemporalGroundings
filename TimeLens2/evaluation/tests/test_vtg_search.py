import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import compare_vtg_search
import run_vtg_search
from vlmeval.omtg_search import Window, strict_embedding_policy


class StrictBudgetPolicyTest(unittest.TestCase):
    def test_policy_coalesces_and_never_overflows(self):
        windows = [Window(index * 40, (index + 1) * 40) for index in range(100)]
        for budget in (64, 128):
            routed, policy = strict_embedding_policy(windows, budget)
            self.assertLessEqual(policy['router_frames'] + policy['local_budget'], budget)
            self.assertGreaterEqual(policy['local_budget'], 2 * policy['selected_window_count'])
            self.assertEqual(len(routed), policy['window_count'])
        self.assertGreater(strict_embedding_policy(windows, 64)[1]['coalesced_window_count'], 0)

    def test_short_video_policy_preserves_one_window(self):
        windows, policy = strict_embedding_policy([Window(0, 12)], 64)
        self.assertEqual(windows, [Window(0, 12)])
        self.assertEqual(policy['selected_window_count'], 1)


class PromptTest(unittest.TestCase):
    def test_controlled_prompt_is_cardinality_aware(self):
        multi = run_vtg_search.build_prompt(
            run_vtg_search.BENCHMARKS['vue-tr-v2'],
            'waves',
            20,
            local=True,
            mode='controlled',
        )
        single = run_vtg_search.build_prompt(
            run_vtg_search.BENCHMARKS['ego4d-nlq-v2'],
            'opens door',
            20,
            local=False,
            mode='controlled',
        )
        self.assertIn('every disjoint', multi)
        self.assertIn('relative to the start of this clip', multi)
        self.assertIn('single best', single)
        self.assertIn('relative to the full video', single)

    def test_native_prompt_adds_coordinate_contract(self):
        prompt = run_vtg_search.build_prompt(
            run_vtg_search.BENCHMARKS['qvhighlights-timelens'],
            'runs',
            12.5,
            local=True,
            mode='native-style',
        )
        self.assertIn('Please find the visual event', prompt)
        self.assertIn('relative to the start of this clip', prompt)


class DatasetLoaderTest(unittest.TestCase):
    @staticmethod
    def _touch_video(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def test_qvhighlights_loader(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._touch_video(root / 'videos/qvhighlights/v1.mp4')
            (root / 'qvhighlights-timelens.json').write_text(json.dumps({
                'v1': {'duration': 10, 'spans': [[1, 2]], 'queries': ['event.']}
            }), encoding='utf-8')
            rows, metadata = run_vtg_search.load_benchmark(
                run_vtg_search.BENCHMARKS['qvhighlights-timelens'], root, 0
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['query'], 'event')
        self.assertEqual(metadata['coverage'], 1.0)

    def test_vue_loader_reports_missing_video_coverage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._touch_video(root / 'videos/a.mp4')
            annotation = [
                {'query_id': 0, 'video_id': 'a', 'duration': 10, 'query': 'x', 'gt': [[1, 2]]},
                {'query_id': 1, 'video_id': 'b', 'duration': 10, 'query': 'y', 'gt': [[3, 4]]},
                {
                    'query_id': 2, 'video_id': 'a', 'duration': 10, 'query': 'audio',
                    'gt': [[5, 6]], 'query_modality': 'audio',
                },
            ]
            (root / 'VUE-TRv2_ground_truth.json').write_text(
                json.dumps(annotation), encoding='utf-8'
            )
            rows, metadata = run_vtg_search.load_benchmark(
                run_vtg_search.BENCHMARKS['vue-tr-v2'], root, 0
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(metadata['coverage'], 0.5)

    def test_momentseeker_loader_uses_text_query_subset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._touch_video(root / 'videos/m1.mp4')
            (root / 't2v.json').write_text(json.dumps([{
                'src_video_path': 'm1.mp4',
                'qry_text': 'find action',
                'answering_time_interval': [[2, 3]],
                'duration': 20,
                'task': 'event',
            }]), encoding='utf-8')
            rows, _ = run_vtg_search.load_benchmark(
                run_vtg_search.BENCHMARKS['momentseeker'], root, 0
            )
        self.assertEqual(rows[0]['targets'], [[2.0, 3.0]])
        self.assertEqual(rows[0]['group'], 'event')

    def test_ego4d_loader_uses_external_video_root(self):
        with TemporaryDirectory() as annotations, TemporaryDirectory() as videos:
            root = Path(annotations)
            video_root = Path(videos)
            self._touch_video(video_root / 'e1.mp4')
            (root / 'ego4d_nlq_val_v2.jsonl').write_text(json.dumps({
                'video_id': 'e1',
                'query_id': 'q1',
                'query': 'Query text: pick up cup',
                'duration': 30,
                'timestamps': [[4, 6]],
                'query_type': 'nlq',
            }) + '\n', encoding='utf-8')
            with patch.dict(os.environ, {'EGO4D_NLQ_V2_VIDEOS_DIR': str(video_root)}):
                rows, _ = run_vtg_search.load_benchmark(
                    run_vtg_search.BENCHMARKS['ego4d-nlq-v2'], root, 0
                )
        self.assertEqual(rows[0]['id'], 'q1')
        self.assertEqual(rows[0]['query'], 'pick up cup')

    def test_ego4d_loader_accepts_official_nested_annotations(self):
        with TemporaryDirectory() as annotations, TemporaryDirectory() as videos:
            root = Path(annotations)
            video_root = Path(videos)
            self._touch_video(video_root / 'clip1.mp4')
            (root / 'nlq_val.json').write_text(json.dumps({
                'videos': [{
                    'video_uid': 'video1',
                    'clips': [{
                        'clip_uid': 'clip1',
                        'clip_start_sec': 100,
                        'clip_end_sec': 130,
                        'annotations': [{
                            'annotation_uid': 'a1',
                            'language_queries': [{
                                'query': 'pick up cup',
                                'clip_start_sec': 4,
                                'clip_end_sec': 6,
                                'video_start_sec': 104,
                                'video_end_sec': 106,
                                'template': 'what did I pick up',
                            }],
                        }],
                    }],
                }],
            }), encoding='utf-8')
            with patch.dict(os.environ, {'EGO4D_NLQ_V2_VIDEOS_DIR': str(video_root)}):
                rows, _ = run_vtg_search.load_benchmark(
                    run_vtg_search.BENCHMARKS['ego4d-nlq-v2'], root, 0
                )
        self.assertEqual(rows[0]['video'], 'clip1')
        self.assertEqual(rows[0]['duration'], 30.0)
        self.assertEqual(rows[0]['targets'], [[4.0, 6.0]])


class ScheduleExecutionTest(unittest.TestCase):
    class FakeGrounder:
        def generate(self, message, dataset=None):
            return '[[1, 2], [4, 5]]'

    @staticmethod
    def _fake_extract(video_path, timestamps, destination, max_width=336):
        return [f'/tmp/frame-{index}.jpg' for index in range(len(timestamps))]

    @staticmethod
    def _row():
        return {
            'id': 'sample', 'video': 'fake', 'video_path': '/tmp/fake.mp4',
            'duration': 100.0, 'query': 'event', 'targets': [[1, 2]], 'group': '',
        }

    @staticmethod
    def _route():
        return {
            'duration': 100.0,
            'windows': [{'start': 0, 'end': 50}, {'start': 48, 'end': 100}],
            'scores': [0.9, 0.5],
            'selected': [0, 1],
            'bypass': False,
            'router_frames': 8,
            'local_budget': 56,
            'router_recall': 1.0,
            'telemetry': {
                'gpu_seconds': 0.5, 'wall_seconds': 0.6, 'peak_vram_bytes': 100,
                'embedding_calls': 3,
            },
        }

    def test_embedding_multispan_maps_and_accounts(self):
        with patch.object(run_vtg_search, 'extract_frames', side_effect=self._fake_extract):
            result = run_vtg_search.execute_setting(
                model=self.FakeGrounder(),
                spec=run_vtg_search.BENCHMARKS['vue-tr-v2'],
                row=self._row(),
                schedule='embedding-window-local',
                prompt_mode='controlled',
                budget=64,
                route=self._route(),
                cache_root=Path('/tmp/cache'),
                frame_width=336,
                native_frame_cap=512,
            )
        self.assertEqual(result['total_frames'], 64)
        self.assertEqual(result['budget_overflow_frames'], 0)
        self.assertEqual(result['grounder_calls'], 2)
        self.assertEqual(result['router_calls'], 3)
        self.assertEqual(result['model_calls'], 5)
        self.assertEqual(
            result['prediction'],
            [[1.0, 2.0], [4.0, 5.0], [49.0, 50.0], [52.0, 53.0]],
        )

    def test_single_span_uses_highest_router_score(self):
        with patch.object(run_vtg_search, 'extract_frames', side_effect=self._fake_extract):
            result = run_vtg_search.execute_setting(
                model=self.FakeGrounder(),
                spec=run_vtg_search.BENCHMARKS['ego4d-nlq-v2'],
                row=self._row(),
                schedule='embedding-window-local',
                prompt_mode='controlled',
                budget=64,
                route=self._route(),
                cache_root=Path('/tmp/cache'),
                frame_width=336,
                native_frame_cap=512,
            )
        self.assertEqual(result['prediction'], [[1.0, 2.0]])

    def test_native_reference_is_capped(self):
        with patch.object(run_vtg_search, 'extract_frames', side_effect=self._fake_extract):
            result = run_vtg_search.execute_setting(
                model=self.FakeGrounder(),
                spec=run_vtg_search.BENCHMARKS['momentseeker'],
                row={**self._row(), 'duration': 1000.0},
                schedule=run_vtg_search.NATIVE_SCHEDULE,
                prompt_mode='controlled',
                budget=512,
                route=None,
                cache_root=Path('/tmp/cache'),
                frame_width=336,
                native_frame_cap=512,
            )
        self.assertEqual(result['total_frames'], 512)


class StatisticsAndReportTest(unittest.TestCase):
    def test_cluster_bootstrap_and_holm_are_deterministic(self):
        first = run_vtg_search._cluster_bootstrap(
            np.asarray([1.0, 0.8, 0.7]),
            np.asarray([0.1, 0.2, 0.3]),
            ['a', 'a', 'b'],
            samples=500,
            seed=7,
        )
        second = run_vtg_search._cluster_bootstrap(
            np.asarray([1.0, 0.8, 0.7]),
            np.asarray([0.1, 0.2, 0.3]),
            ['a', 'a', 'b'],
            samples=500,
            seed=7,
        )
        self.assertEqual(first, second)
        rows = [
            {'p_value': 0.01, 'ci95_low': 0.1, 'ci95_high': 1.0},
            {'p_value': 0.04, 'ci95_low': 0.1, 'ci95_high': 1.0},
        ]
        run_vtg_search._holm_adjust(rows)
        self.assertEqual(rows[0]['holm_p_value'], 0.02)
        self.assertEqual(rows[1]['holm_p_value'], 0.04)

    def test_combined_verdict_requires_two_long_benchmarks(self):
        positive = {
            'comparisons': [{
                'significant_positive': True,
                'significant_negative': False,
            }]
        }
        neutral = {
            'comparisons': [{
                'significant_positive': False,
                'significant_negative': False,
            }]
        }
        result = compare_vtg_search.compile_result({
            'vue-tr-v2': positive,
            'momentseeker': positive,
            'ego4d-nlq-v2': neutral,
        })
        self.assertTrue(result['broad_long_form_transfer_supported'])
        incomplete = compare_vtg_search.compile_result({'vue-tr-v2': positive})
        self.assertIsNone(incomplete['broad_long_form_transfer_supported'])

    def test_evaluate_phase_writes_all_report_formats(self):
        row = {
            'id': 'sample', 'video': 'video', 'targets': [[1.0, 2.0]], 'group': 'event',
        }
        base = {
            **row,
            'duration': 10.0,
            'query': 'event',
            'grounder_frames': 64,
            'embedding_frames': 0,
            'total_frames': 64,
            'unused_frames': 0,
            'budget_overflow_frames': 0,
            'grounder_calls': 1,
            'router_calls': 0,
            'model_calls': 1,
            'model_seconds': 0.1,
            'wall_seconds': 0.2,
            'peak_vram_bytes': 100,
            'router_recall': None,
        }
        records = []
        for budget in (64, 128):
            records.extend([
                {
                    **base,
                    'schedule': 'embedding-window-local',
                    'budget': budget,
                    'prompt_mode': 'controlled',
                    'prediction': [[1.0, 2.0]],
                },
                {
                    **base,
                    'schedule': 'uniform-one-shot',
                    'budget': budget,
                    'prompt_mode': 'controlled',
                    'prediction': [[5.0, 6.0]],
                },
            ])
        args = SimpleNamespace(
            budgets=[64, 128],
            prompt_modes=['controlled', 'native-style'],
            native_frame_cap=512,
            bootstrap_samples=50,
            bootstrap_seed=7,
            model='grounder',
            embedding_model='router',
        )
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'predictions.jsonl').write_text(
                ''.join(json.dumps(record) + '\n' for record in records),
                encoding='utf-8',
            )
            result = run_vtg_search.evaluate_phase(
                args,
                run_vtg_search.BENCHMARKS['momentseeker'],
                [row],
                run_dir,
            )
            self.assertTrue((run_dir / 'summary.json').is_file())
            self.assertTrue((run_dir / 'summary.csv').is_file())
            self.assertTrue((run_dir / 'comparisons.csv').is_file())
            self.assertTrue((run_dir / 'prompt_comparisons.csv').is_file())
            self.assertTrue((run_dir / 'report.md').is_file())
        self.assertEqual([comparison['delta'] for comparison in result['comparisons']], [100, 100])


if __name__ == '__main__':
    unittest.main()
