import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import compare_omtg_inference as comparison


class BootstrapTest(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        candidate = np.asarray([1.0, 2.0, 3.0, 4.0])
        reference = np.asarray([0.0, 1.0, 1.0, 2.0])
        first = comparison.paired_bootstrap_interval(candidate, reference, 1_000, 42)
        second = comparison.paired_bootstrap_interval(candidate, reference, 1_000, 42)
        self.assertEqual(first, second)
        self.assertLess(first[0], 1.5)
        self.assertGreater(first[1], 1.5)

    def test_paired_bootstrap_rejects_unpaired_arrays(self):
        with self.assertRaisesRegex(ValueError, 'same shape'):
            comparison.paired_bootstrap_interval(
                np.asarray([1.0, 2.0]),
                np.asarray([1.0]),
                100,
                1,
            )

    def test_cluster_bootstrap_is_deterministic(self):
        candidate = np.asarray([1.0, 2.0, 4.0])
        reference = np.asarray([0.0, 0.0, 1.0])
        first = comparison.paired_cluster_bootstrap_interval(
            candidate, reference, ['a', 'a', 'b'], 100, 7
        )
        second = comparison.paired_cluster_bootstrap_interval(
            candidate, reference, ['a', 'a', 'b'], 100, 7
        )
        self.assertEqual(first, second)


class ExperimentLoadingTest(unittest.TestCase):
    @staticmethod
    def write_baseline(directory: Path, duplicate: bool = False) -> Path:
        records = [
            {
                'id': 0,
                'question': 'Find the video segment for event A and determine its start and end seconds.',
                'answer': '[[1, 2]]',
                'prompt_mode': 'official',
                'response': 'Start: 1, End: 2',
            },
            {
                'id': 1,
                'question': 'Find the video segment for event B and determine its start and end seconds.',
                'answer': '[[3, 4]]',
                'prompt_mode': 'official',
                'response': '[[3, 4]]',
            },
        ]
        if duplicate:
            records.append(dict(records[0]))
        prediction_path = directory / 'predictions.jsonl'
        prediction_path.write_text(
            ''.join(json.dumps(record) + '\n' for record in records),
            encoding='utf-8',
        )
        metrics = comparison.scaled_metrics([[1.0, 2.0]], '[[1, 2]]')
        summary = {
            'model': 'example/model',
            'results': [{
                'schedule': 'paper-2fps-official',
                'complete': True,
                'samples': 2,
                **metrics,
            }],
        }
        summary_path = directory / 'summary.json'
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        return summary_path

    def test_baseline_rescores_raw_response(self):
        with TemporaryDirectory() as directory:
            experiment = comparison.load_experiment(
                self.write_baseline(Path(directory)),
                'paper-style-2fps',
            )
        key = 'paper-2fps-official'
        self.assertEqual(experiment.prediction_counts[key], {0: 1, 1: 1})
        self.assertEqual(experiment.per_query[key][0]['EtF1'], 100.0)

    def test_duplicate_prediction_is_rejected(self):
        with TemporaryDirectory() as directory:
            summary_path = self.write_baseline(Path(directory), duplicate=True)
            with self.assertRaisesRegex(ValueError, 'Duplicate prediction'):
                comparison.load_experiment(summary_path, 'paper-style-2fps')

    def test_stale_summary_is_rejected(self):
        with TemporaryDirectory() as directory:
            summary_path = self.write_baseline(Path(directory))
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            summary['results'][0]['EtF1'] = 0.0
            summary_path.write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'does not match rescored value'):
                comparison.load_experiment(summary_path, 'paper-style-2fps')


class MultiModelReportTest(unittest.TestCase):
    def test_result_identifiers_include_model(self):
        metrics = {
            'schedule': 'uniform-one-shot',
            'budget': 64,
            'complete': True,
        }
        first = comparison.normalized_result(metrics, 'fixed-frame', 'first/model')
        second = comparison.normalized_result(metrics, 'fixed-frame', 'second/model')
        self.assertNotEqual(first['result_id'], second['result_id'])
        self.assertEqual(first['result'], second['result'])

    def test_model_sorting_prefers_timelens_then_qwen(self):
        models = [
            'Qwen/Qwen3-VL-4B-Instruct',
            'other/model',
            'MCG-NJU/TimeLens2-4B',
        ]
        self.assertEqual(
            sorted(models, key=comparison.model_sort_key),
            [
                'MCG-NJU/TimeLens2-4B',
                'Qwen/Qwen3-VL-4B-Instruct',
                'other/model',
            ],
        )

    def test_output_paths_preserve_public_filenames(self):
        paths = comparison.output_paths(Path('/tmp/results'))
        self.assertEqual(paths['markdown'].name, 'comparison.md')
        self.assertEqual(paths['results_csv'].name, 'omtg_inference_results.csv')
        self.assertEqual(paths['deltas_csv'].name, 'omtg_inference_deltas.csv')


if __name__ == '__main__':
    unittest.main()
