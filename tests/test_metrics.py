import unittest
from metrics import score_query,summarize
from review_answers import evaluate_rows
from evaluate_retrieval import labels_from_documents,run

class TestMetrics(unittest.TestCase):
    def test_first_relevant_second(self):
        result=score_query(['x','a','b'],{'a','b'},3)
        self.assertEqual(result,{'recall':1.0,'hit':1.0,'rr':0.5})
    def test_unmatched(self):
        self.assertEqual(score_query(['x','y'],{'z'},2)['rr'],0)
    def test_top_k(self):
        self.assertEqual(score_query(['x','y','a'],{'a'},2)['hit'],0)
    def test_empty_gold_invalid(self):
        with self.assertRaises(ValueError):score_query(['x'],set(),5)
    def test_duplicate_retrieved_invalid(self):
        with self.assertRaises(ValueError):score_query(['x','x'],{'x'},5)
    def test_average(self):
        a=score_query(['a'],{'a'},5);b=score_query(['x'],{'a'},5)
        self.assertEqual(summarize([a,b])['mrr_at_k'],0.5)
    def test_labels(self):
        self.assertEqual(labels_from_documents([{'id':'abc','metadata':{'topic':'dict'}}])['dict'],{'abc'})
    def test_manual_review_empty(self):
        self.assertEqual(evaluate_rows([{'correctness':'','faithfulness':''}])['reviewed'],0)
    def test_manual_review_one(self):
        out=evaluate_rows([{'correctness':'1','faithfulness':'0'}])
        self.assertEqual(out['faithfulness_rate'],0)
    def test_adapter_fake(self):
        class Fake:
            documents=[{'id':'a','metadata':{'topic':'dict'}},{'id':'b','metadata':{'topic':'list'}}]
            def search(self,q,mode,top_k,candidate_k):
                return [{'id':'b'},{'id':'a'}] if mode=='vector' else [{'id':'a'},{'id':'b'}]
        rows,summary=run(Fake(),[{'id':'q','question':'lookup','relevant_topics':['dict']}],['vector','bm25'],k=2,candidate_k=3)
        self.assertEqual(rows[0]['rr'],0.5)
        self.assertEqual(rows[1]['rr'],1)
        self.assertEqual(summary['bm25']['mrr_at_k'],1)
if __name__=='__main__':unittest.main()
