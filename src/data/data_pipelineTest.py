import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from data.data_pipeline import DataPipeline  # Assuming this is the class containing _save_results

class DataPipelineTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = DataPipeline()
        self.pipeline.stats = {
            "success": False,
            "errors": [],
            "end_time": None
        }
        
    @patch('data.data_pipeline.MySQLWriter')
    def test_save_results_successful_database_writes(self, mock_writer):
        """Test successful database writes"""
        # Setup mock writer
        mock_instance = mock_writer.return_value
        mock_instance.write_predictions.return_value = True
        mock_instance.write_history.return_value = True
        
        # Create test data
        predictions = pd.DataFrame({
            'sku': ['SKU001', 'SKU002'],
            'price': [10.99, 20.50],
            'date': [datetime.now(), datetime.now()],
            'confidence': [0.95, 0.92]
        })
        
        raw_data = pd.DataFrame({
            'sku_id': ['SKU001', 'SKU002'],
            'date': [datetime.now().date(), datetime.now().date()],
            'price': [10.99, 20.50],
            'is_promotion': [False, True]
        })
        
        # Call the method
        self.pipeline._save_results(predictions, raw_data)
        
        # Assertions
        mock_instance.write_predictions.assert_called_once_with(predictions)
        mock_instance.write_history.assert_called_once_with(raw_data)
        self.assertTrue(self.pipeline.stats["success"])
        self.assertIsNotNone(self.pipeline.stats["end_time"])
        self.assertEqual(len(self.pipeline.stats["errors"]), 0)

if __name__ == '__main__':
    unittest.main()