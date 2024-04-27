import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import mdtraj as md
from pages import Feature_Extraction

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.trajectory = md.load('tests/data/test_trajectory.xtc', top='tests/data/test_topology.pdb')

    @patch('1_Feature_Extraction.st.success')
    def test_compute_rmsd(self, mock_success):
        Feature_Extraction.compute_rmsd(self.trajectory)
        mock_success.assert_called_once_with("Successfully computed rmsd")

    @patch('1_Feature_Extraction.st.success')
    def test_compute_sasa(self, mock_success):
        Feature_Extraction.compute_sasa(self.trajectory)
        mock_success.assert_called_once_with("Successfully computed SaSa")

    @patch('1_Feature_Extraction.st.success')
    def test_compute_rog(self, mock_success):
        Feature_Extraction.compute_rog(self.trajectory)
        mock_success.assert_called_once_with("Successfully computed Radius of Gyration")

    @patch('1_Feature_Extraction.st.success')
    def test_compute_h_bonds(self, mock_success):
        Feature_Extraction.compute_h_bonds(self.trajectory)
        mock_success.assert_called_once_with("Successfully computed Hydrogen Bonds")

    @patch('1_Feature_Extraction.st.success')
    def test_compute_native_contacts(self, mock_success):
        Feature_Extraction.compute_native_contacts(self.trajectory)
        mock_success.assert_called_once_with("Successfully computed fraction of native contacts that determine protein folding")

    @patch('1_Feature_Extraction.st.success')
    @patch('1_Feature_Extraction.st.file_uploader')
    @patch('1_Feature_Extraction.st.button')
    def test_main(self, mock_button, mock_file_uploader, mock_success):
        mock_button.return_value = True
        mock_file_uploader.return_value = MagicMock()
        mock_file_uploader.return_value.getvalue.return_value = b"test"
        Feature_Extraction.main()
        mock_success.assert_called_once_with("Successfully loaded the trajectory and we are ready to start analysis")

if __name__ == '__main__':
    unittest.main()