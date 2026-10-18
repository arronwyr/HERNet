import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'evaluation/cmc.py'
spec=importlib.util.spec_from_file_location('cmc',path)
cmc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cmc)

class CMCTests(unittest.TestCase):
    def test_subject_retrieval(self):
        scores=np.array([[9,8,2],[4,3,7]],dtype=np.int8)
        ranks=cmc.first_relevant_ranks(scores,['1_x','2_x'],['1_a','1_b','2_a'],'molf')
        np.testing.assert_array_equal(ranks,[1,1])
        np.testing.assert_array_equal(cmc.curve(ranks,3),[1,1,1])

    def test_ties_follow_numpy_not_optimistic_rank(self):
        scores=np.array([[0,0,0]],dtype=np.int8)
        gallery=['1_db36_RR_1_500_0','2_db36_RR_1_500_0','3_db36_RR_1_500_0']
        probe=['1_db1_RR_1_500_0']
        expected=int(np.flatnonzero(np.argsort(scores[0])[::-1]==0)[0])+1
        self.assertEqual(int(cmc.first_relevant_ranks(scores,probe,gallery,'must')[0]),expected)

if __name__=='__main__':unittest.main()
