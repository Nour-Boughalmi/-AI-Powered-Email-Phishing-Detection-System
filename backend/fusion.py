import pandas as pd
import numpy as np
from preprocessing import EmailPreprocessor
import joblib

class FeatureFusion:
    """Fusionne les features NLP avec les features traditionnelles"""
    
    def __init__(self):
        self.preprocessor = EmailPreprocessor()
        
    def extract_all_features(self, email_text):
        """Extrait toutes les features d'un email brut"""
        
        # Créer un DataFrame temporaire
        temp_df = pd.DataFrame({
            'subject': [''] * len(email_text) if isinstance(email_text, list) else [''],
            'body': email_text if isinstance(email_text, list) else [email_text],
            'label': [0]  # Label placeholder
        })
        
        if isinstance(email_text, str):
            temp_df = pd.DataFrame({
                'subject': [''],
                'body': [email_text],
                'label': [0]
            })
        
        # Préparer les features
        features = self.preprocessor.prepare_features(temp_df)
        
        return features

# Test
if __name__ == "__main__":
    fusion = FeatureFusion()
    
    test_email = "URGENT: Your account has been compromised! Verify now at http://bit.ly/verify"
    features = fusion.extract_all_features(test_email)
    
    print("Features extraites:")
    print(features.to_string())