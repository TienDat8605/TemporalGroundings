import io
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import colab_remote_job  # noqa: E402


class CheckpointExtractionTest(unittest.TestCase):
    def test_checkpoint_overlays_existing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'checkpoint.tar.gz'
            destination = root / 'outputs'
            destination.mkdir()
            (destination / 'existing.jsonl').write_text('old\n', encoding='utf-8')

            payload = b'{"id": 1}\n'
            with tarfile.open(archive, 'w:gz') as bundle:
                member = tarfile.TarInfo('smoke/predictions.jsonl')
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))

            colab_remote_job.safe_extract_into(archive, destination)

            self.assertEqual(
                (destination / 'smoke' / 'predictions.jsonl').read_bytes(),
                payload,
            )
            self.assertEqual(
                (destination / 'existing.jsonl').read_text(encoding='utf-8'),
                'old\n',
            )

    def test_checkpoint_rejects_parent_traversal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'checkpoint.tar.gz'
            payload = b'unsafe'
            with tarfile.open(archive, 'w:gz') as bundle:
                member = tarfile.TarInfo('../escape.txt')
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                colab_remote_job.safe_extract_into(archive, root / 'outputs')

            self.assertFalse((root / 'escape.txt').exists())


if __name__ == '__main__':
    unittest.main()
