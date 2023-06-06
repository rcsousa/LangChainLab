import unittest
from unittest.mock import patch
from app import generate_tagline


class TestGenerateTagline(unittest.TestCase):
    @patch('app.openai.Completion.create')
    def test_generate_tagline_success(self, mock_create):
        # Mock the API response
        mock_create.return_value.choices[0].text = 'The best ice cream in town!'
        
        # Call the function and check the result
        result = generate_tagline(prompt='Write a tagline for an ice cream shop. ', max_tokens=10)
        self.assertEqual(result, 'The best ice cream in town!')
        
        # Check that the API was called with the correct arguments
        mock_create.assert_called_once_with(engine='text-davinci-003', prompt='Write a tagline for an ice cream shop. ', max_tokens=10)
    
    @patch('app.openai.Completion.create')
    def test_generate_tagline_error(self, mock_create):
        # Mock the API call to raise an exception
        mock_create.side_effect = Exception('API error')
        
        # Call the function and check the result
        result = generate_tagline(prompt='Write a tagline for an ice cream shop. ', max_tokens=10)
        self.assertEqual(result, '')
        
        # Check that the API was called with the correct arguments
        mock_create.assert_called_once_with(engine='text-davinci-003', prompt='Write a tagline for an ice cream shop. ', max_tokens=10)
    
    @patch('app.openai.Completion.create')
    def test_generate_tagline_no_result(self, mock_create):
        # Mock the API response to have no choices
        mock_create.return_value.choices = []
        
        # Call the function and check the result
        result = generate_tagline(prompt='Write a tagline for an ice cream shop. ', max_tokens=10)
        self.assertEqual(result, '')
        
        # Check that the API was called with the correct arguments
        mock_create.assert_called_once_with(engine='text-davinci-003', prompt='Write a tagline for an ice cream shop. ', max_tokens=10)


if __name__ == '__main__':
    unittest.main()