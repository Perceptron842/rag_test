import unittest
from metrics import score_query
from rag_pipeline import build_evidence, run_rag, REFUSAL
from leetcode_11 import max_area


class NeverCall:
    def generate(self, messages):
        raise AssertionError('没有证据时不应该调用 LLM')


class StaticRetriever:
    def __init__(self, hits): self.hits=hits
    def search(self, question, mode='hybrid', top_k=5, candidate_k=10):
        return self.hits[:top_k]


class Day7Tests(unittest.TestCase):
    def test_no_evidence_refusal(self):
        got=run_rag('未知问题',StaticRetriever([]),NeverCall())
        self.assertEqual(got.answer,REFUSAL)
        self.assertEqual(got.status,'insufficient_evidence')

    def test_sources_and_prompt(self):
        hits=[{'id':'a','text':'Python字典通过键取值。','metadata':{'source':'a.pdf','page':2}}]
        ev,src=build_evidence(hits)
        self.assertIn('[资料 1]',ev)
        self.assertEqual(src[0]['source'],'a.pdf，第2页')

    def test_metrics(self):
        self.assertEqual(score_query(['x','a','b'],{'a','b'},2)['recall'],0.5)

    def test_two_pointers(self):
        self.assertEqual(max_area([1,8,6,2,5,4,8,3,7]),49)
        self.assertEqual(max_area([2,3,4,5,18,17,6]),17)

if __name__=='__main__': unittest.main()
