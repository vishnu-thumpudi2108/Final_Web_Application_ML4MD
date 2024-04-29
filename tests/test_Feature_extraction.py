import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd
import your_module  # replace with the actual name of your module

class TestFeatureExtraction(unittest.TestCase):
    @patch('tempfile.NamedTemporaryFile')
    @patch('your_module.md')
    @patch('your_module.st')
    @patch('your_module.pd')
    @patch('your_module.compute_rmsd')
    @patch('your_module.compute_rmsf')
    @patch('your_module.compute_sasa')
    @patch('your_module.compute_rog')
    @patch('your_module.compute_h_bonds')
    @patch('your_module.compute_native_contacts')
    def test_feature_extraction(self, mock_nc, mock_hb, mock_rog, mock_sasa, mock_rmsf, mock_rmsd, mock_pd, mock_st, mock_md, mock_tempfile):
        # Arrange
        mock_tempfile.return_value.__enter__.return_value.name = 'tempfile'
        mock_md.load.return_value = MagicMock(n_frames=10, n_atoms=20, n_residues=30)
        mock_rmsd.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_rmsf.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_sasa.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_rog.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Act
        your_module.your_function()  # replace with the actual name of your function

        # Assert
        mock_md.load.assert_called_once_with('tempfile', top='tempfile')
        mock_st.success.assert_called_once_with("Successfully loaded the trajectory and we are ready to start analysis")
        mock_pd.DataFrame.assert_called_once_with({'Frame': range(1, 11)})
        mock_rmsd.assert_called_once()
        mock_rmsf.assert_called_once()
        mock_sasa.assert_called_once()
        mock_rog.assert_called_once()
        mock_hb.assert_called_once()
        mock_nc.assert_called_once()

if __name__ == '__main__':
    unittest.main()