import unittest
from rag_pipeline import (
    REFUSAL, MockGenerator, build_evidence, build_messages, run_rag,
)

GOOD = {'id':'id1','text':'Python字典通过 key 查找对应的值。',
        'metadata':{'source':'test.pdf','page':3,'chunk_index':0},'cosine_similarity':0.75}


class FakeRetriever:
    def __init__(self, hits=None, sim=None):
        self.hits = hits if hits is not None else [GOOD]
        self.sim = sim
        self.calls = []

    def search(self, question, mode='hybrid', top_k=5, candidate_k=10):
        self.calls.append((mode, top_k, candidate_k))
        if mode == 'vector' and self.sim is not None:
            return [{**GOOD,'cosine_similarity':self.sim}]
        return self.hits[:top_k]


class SpyGenerator:
    def __init__(self): self.called = 0; self.messages = None
    def generate(self, messages):
        self.called += 1
        self.messages = messages
        return '根据资料，可以用 data[key] 访问字典中的值。[1]'


class PipelineTests(unittest.TestCase):
    def test_prompt_has_question_and_reference(self):
        evidence, sources = build_evidence([GOOD])
        self.assertIn('[资料 1]', evidence)
        self.assertIn('test.pdf，第3页', evidence)
        self.assertEqual(sources[0]['number'], 1)
        messages = build_messages('如何查字典？', evidence)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('如何查字典？', messages[1]['content'])
        self.assertIn('不可信文本', messages[0]['content'])

    def test_success(self):
        generator = SpyGenerator()
        result = run_rag('如何查字典？', FakeRetriever(), generator)
        self.assertEqual(result.status,'answered')
        self.assertEqual(generator.called,1)
        self.assertIn('[1]',result.answer)

    def test_empty_retrieval_refuses_without_llm(self):
        generator = SpyGenerator()
        result = run_rag('天气？', FakeRetriever(hits=[]), generator)
        self.assertEqual(result.status,'insufficient_evidence')
        self.assertEqual(result.answer,REFUSAL)
        self.assertEqual(generator.called,0)

    def test_low_similarity_refuses_without_llm(self):
        generator = SpyGenerator()
        result = run_rag('天气？', FakeRetriever(sim=0.05), generator, min_cosine=0.35)
        self.assertEqual(result.status,'insufficient_evidence')
        self.assertEqual(generator.called,0)

    def test_mock_is_explicit(self):
        result = run_rag('如何查字典？', FakeRetriever(), MockGenerator())
        self.assertEqual(result.status,'mock')
        self.assertIn('没有调用 LLM', result.answer)

    def test_evidence_limits_and_dedup(self):
        text, sources = build_evidence([GOOD, GOOD], per_chunk_chars=12, total_chars=20)
        self.assertEqual(len(sources),1)
        self.assertLessEqual(len(sources[0]['excerpt']),12)

    def test_empty_answer_raises(self):
        class EmptyGen:
            def generate(self, msgs): return ''
        with self.assertRaisesRegex(RuntimeError,'empty'):
            run_rag('Hi', FakeRetriever(), EmptyGen())

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            run_rag('   ', FakeRetriever(), MockGenerator())
        with self.assertRaises(ValueError):
            run_rag('Q', FakeRetriever(), MockGenerator(),top_k=8,candidate_k=2)


if __name__ == '__main__':
    unittest.main()
