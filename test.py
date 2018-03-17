import unittest

from pathlib import Path
from src.face_detection import cropped_images
from src.ocr import ada_ocr_v2

DATA_DIR =  Path(__file__).resolve().parent/'data'

class OCRTestCases(unittest.TestCase):
    '''Test Cases for ocr.py'''

    def test_correct_bib(self):
        cropped = cropped_images(image_path=DATA_DIR/'y001.jpg', see_full=False)
        predicted = ada_ocr_v2(images=cropped, debug=False)
        self.assertIn('878', predicted)

if __name__ == '__main__':
    unittest.main()
