"""Small shape/finite-output checks; no dataset or trained weights required."""
import unittest
import torch
from hernet import HERNet

class ModelTests(unittest.TestCase):
    def test_variable_shapes_and_constant_input(self):
        torch.set_num_threads(2)
        model=HERNet().eval()
        with torch.inference_mode():
            for h,w in [(32,32),(37,49),(64,96)]:
                y=model.enhance(torch.ones(1,h,w))
                self.assertEqual(tuple(y.shape),(1,h,w))
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(((y>=0)&(y<=1)).all())

if __name__=='__main__':unittest.main()
