"""Index lifecycle regression tests; no model download or LLM calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from retriever import COLLECTION_NAME, build_index


class IndexTests(unittest.TestCase):
    def setUp(self):
        try:
            import chromadb
            import numpy as np
        except ImportError:
            self.skipTest('Chroma and NumPy are required for index integration tests')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = str(Path(self.temp.name) / 'db')
        self.client = chromadb.PersistentClient(path=self.db)
        # Release database handles before TemporaryDirectory cleanup on Windows.
        self.addCleanup(getattr(self.client, 'close', self.client._system.stop))
        self.input = Path(__file__).resolve().parents[1] / 'data' / 'sample_chunks.json'
        self.encoder = Mock()
        self.encoder.encode.side_effect = lambda texts, **kwargs: np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1)
        )
        client_patch = patch('retriever.get_client', return_value=self.client)
        encoder_patch = patch('retriever.get_encoder', return_value=self.encoder)
        client_patch.start()
        encoder_patch.start()
        self.addCleanup(client_patch.stop)
        self.addCleanup(encoder_patch.stop)

    def test_reset_builds_missing_collection(self):
        build_index(self.input, db_path=self.db, reset=True, model_name='test-model')
        collection = self.client.get_collection(COLLECTION_NAME, embedding_function=None)
        self.assertGreater(collection.count(), 0)
        self.assertEqual(collection.metadata['model_name'], 'test-model')

    def test_reset_replaces_existing_collection_and_preserves_other_collections(self):
        old = self.client.create_collection(COLLECTION_NAME, embedding_function=None)
        old.upsert(ids=['obsolete'], embeddings=[[0.0, 1.0, 0.0]], documents=['old'])
        other = self.client.create_collection('unrelated', embedding_function=None)
        other.upsert(ids=['keep'], embeddings=[[0.0, 1.0, 0.0]], documents=['keep'])

        build_index(self.input, db_path=self.db, reset=True, model_name='test-model')

        collection = self.client.get_collection(COLLECTION_NAME, embedding_function=None)
        self.assertGreater(collection.count(), 0)
        self.assertEqual(collection.get(ids=['obsolete'])['ids'], [])
        self.assertEqual(self.client.get_collection('unrelated').get()['ids'], ['keep'])

    def test_reset_does_not_hide_other_errors(self):
        with patch.object(self.client, 'delete_collection', side_effect=RuntimeError('disk failure')):
            with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                build_index(self.input, db_path=self.db, reset=True)
        self.encoder.encode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
