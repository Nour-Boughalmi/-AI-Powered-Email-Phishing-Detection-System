import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
import tldextract

# Télécharger les ressources NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EmailPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Nettoie le texte d'un email"""
        if not isinstance(text, str):
            text = str(text)
        
        # Mettre en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Supprimer les emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Supprimer la ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Supprimer les chiffres
        text = re.sub(r'\d+', '', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features_text(self, text):
        """Extrait des features textuelles basiques"""
        features = {}
        
        # Longueur
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Mots suspects de phishing
        suspicious_words = ['urgent', 'verify', 'account', 'password', 'click', 'login', 
                           'security', 'update', 'confirm', 'bank', 'paypal', 'amazon',
                           'suspended', 'locked', 'alert', 'immediate', 'action', 'required',
                           'limited', 'unusual', 'compromised', 'verify', 'information']
        
        features['suspicious_word_count'] = sum(1 for word in suspicious_words 
                                                 if word in text.lower())
        
        # Ponctuation
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Majuscules (dans le texte original)
        features['upper_ratio'] = 0  # Sera calculé sur le texte original
        
        return features
    
    def extract_url_features(self, text):
        """Extrait des features des URLs dans l'email"""
        # Trouver toutes les URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        features = {
            'url_count': len(urls),
            'has_url': 1 if len(urls) > 0 else 0,
            'suspicious_url': 0,
            'has_shortened_url': 0,
            'has_ip_url': 0
        }
        
        shortened_domains = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'shorturl']
        
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc
                
                # Vérifier les raccourcisseurs
                if any(short in domain.lower() for short in shortened_domains):
                    features['has_shortened_url'] = 1
                    features['suspicious_url'] = 1
                
                # Vérifier si l'URL contient une IP
                if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
                    features['has_ip_url'] = 1
                    features['suspicious_url'] = 1
                
                # Vérifier les noms de domaine suspects
                ext = tldextract.extract(url)
                if 'login' in ext.subdomain or 'verify' in ext.subdomain or 'secure' in ext.subdomain:
                    features['suspicious_url'] = 1
                    
            except:
                pass
        
        return features
    
    def prepare_features(self, df):
        """Prépare toutes les features pour le modèle"""
        print("📊 Préparation des features...")
        
        # Nettoyer le texte
        print("  - Nettoyage du texte...")
        df['clean_body'] = df['body'].apply(self.clean_text)
        df['clean_subject'] = df['subject'].apply(self.clean_text)
        df['full_text'] = df['clean_subject'] + ' ' + df['clean_body']
        
        # Features textuelles
        print("  - Extraction features textuelles...")
        text_features = df['full_text'].apply(self.extract_features_text).apply(pd.Series)
        
        # Features URLs
        print("  - Extraction features URLs...")
        url_features_body = df['body'].apply(self.extract_url_features).apply(pd.Series)
        url_features_subject = df['subject'].apply(self.extract_url_features).apply(pd.Series)
        
        # Renommer pour éviter les conflits
        url_features_body = url_features_body.add_prefix('body_')
        url_features_subject = url_features_subject.add_prefix('subject_')
        
        # Calcul du ratio de majuscules sur le texte original
        df['upper_ratio'] = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
        
        # Combiner toutes les features
        all_features = pd.concat([
            text_features,
            url_features_body,
            url_features_subject,
            df[['upper_ratio']]
        ], axis=1)
        
        # Remplacer les NaN par 0
        all_features = all_features.fillna(0)
        
        print(f"  ✅ {len(all_features.columns)} features extraites")
        
        return all_features
    
    def prepare_data(self, csv_path):
        """Charge et prépare les données"""
        print("📂 Chargement des données...")
        df = pd.read_csv(csv_path)
        
        # Vérifier les colonnes
        required_cols = ['body', 'subject', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Colonne '{col}' manquante dans le CSV")
        
        print(f"  - {len(df)} emails chargés")
        print(f"  - Distribution: {df['label'].value_counts().to_dict()}")
        
        # Préparer les features
        features = self.prepare_features(df)
        
        # Target
        y = df['label'].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Split des données:")
        print(f"  - Train: {len(X_train)} samples")
        print(f"  - Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, features.columns.tolist()

# Test rapide
if __name__ == "__main__":
    preprocessor = EmailPreprocessor()
    
    # Créer un petit dataset de test
    test_df = pd.DataFrame({
        'subject': ['Urgent: Verify your account', 'Hello friend', 'Your order has shipped'],
        'body': ['Click here: http://bit.ly/verify now!', 'How are you?', 'Track your package at https://amazon.com'],
        'label': [1, 0, 0]
    })
    
    test_df.to_csv('data/test.csv', index=False)
    
    try:
        X_train, X_test, y_train, y_test, features = preprocessor.prepare_data('data/test.csv')
        print(f"\n✅ Preprocessing fonctionne! Features: {features}")
    except Exception as e:
        print(f"❌ Erreur: {e}")