from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import re

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')
CORS(app)

# Charger le modèle
print("🔧 Chargement du modèle...")
model = None
feature_names = None

try:
    if os.path.exists('models/catboost_model.pkl'):
        model = joblib.load('models/catboost_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        print("✅ Modèle CatBoost chargé avec succès!")
    else:
        print("⚠️ Modèle non trouvé. Utilisation du mode règles simples...")
except Exception as e:
    print(f"⚠️ Erreur chargement: {e}")
    print("   Utilisation du mode règles simples...")

# Fonction d'extraction de features simplifiée (sans dépendance à fusion.py)
def extract_features_simple(email_text):
    """Extrait les features d'un email - version robuste"""
    
    if not isinstance(email_text, str):
        email_text = str(email_text)
    
    text_lower = email_text.lower()
    
    # Features basiques
    features = {
        'length': min(len(email_text), 5000),
        'word_count': len(email_text.split()),
        'exclamation_count': min(email_text.count('!'), 10),
        'question_count': min(email_text.count('?'), 10),
        'upper_ratio': sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1),
    }
    
    # URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, email_text)
    features['url_count'] = len(urls)
    features['has_url'] = 1 if len(urls) > 0 else 0
    
    # URLs suspectes (raccourcisseurs)
    shortened = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'short.url']
    features['has_shortened_url'] = 1 if any(s in text_lower for s in shortened) else 0
    
    # URL avec IP
    features['has_ip_url'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', email_text) else 0
    
    # Mots suspects
    suspicious_words = ['urgent', 'verify', 'account', 'password', 'click', 'login', 
                       'security', 'update', 'confirm', 'bank', 'paypal', 'amazon',
                       'suspended', 'locked', 'alert', 'immediate', 'limited', 
                       'compromised', 'unauthorized', 'blocked', 'warning']
    
    features['suspicious_word_count'] = sum(1 for word in suspicious_words if word in text_lower)
    
    # Mots d'urgence
    urgency_words = ['urgent', 'immediate', 'asap', 'now', 'today', 'hours']
    features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
    
    # Menaces
    threat_words = ['suspended', 'locked', 'closed', 'blocked', 'terminated', 'lost']
    features['threat_score'] = sum(1 for word in threat_words if word in text_lower)
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({'error': 'Aucun email fourni'}), 400
        
        # Extraire les features
        features = extract_features_simple(email_text)
        features_df = pd.DataFrame([features])
        
        # Prédiction avec le modèle ou règles simples
        if model is not None and feature_names is not None:
            try:
                # Aligner les colonnes
                for col in feature_names:
                    if col not in features_df.columns:
                        features_df[col] = 0
                features_df = features_df[feature_names]
                
                # Prédiction
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0]
                
                # SEUIL AJUSTÉ: plus élevé pour réduire les faux positifs
                SEUIL_PHISHING = 0.70
                is_phishing = probability[1] > SEUIL_PHISHING
                
                result = {
                    'is_phishing': bool(is_phishing),
                    'confidence': float(max(probability)),
                    'probability_phishing': float(probability[1]),
                    'probability_legitimate': float(probability[0]),
                    'message': "⚠️ ALERTE: Email suspect détecté!" if is_phishing else "✅ Email légitime",
                    'features': {
                        'url_count': features['url_count'],
                        'suspicious_words': features['suspicious_word_count'],
                        'has_shortened_url': bool(features['has_shortened_url']),
                        'exclamation_marks': features['exclamation_count'],
                        'urgency_score': features['urgency_score'],
                        'threat_score': features['threat_score']
                    }
                }
            except Exception as e:
                print(f"Erreur prédiction: {e}")
                # Fallback sur règles simples
                result = predict_with_rules(features)
        else:
            # Mode règles simples
            result = predict_with_rules(features)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_with_rules(features):
    """Prédiction basée sur des règles simples (fallback)"""
    
    score = 0
    score += features['suspicious_word_count'] * 15
    score += features['has_shortened_url'] * 25
    score += features['has_ip_url'] * 20
    score += features['urgency_score'] * 10
    score += features['threat_score'] * 10
    score += features['exclamation_count'] * 3
    score += features['url_count'] * 5
    
    # Normaliser
    probability = min(score / 100, 0.95)
    is_phishing = probability > 0.5
    
    return {
        'is_phishing': is_phishing,
        'confidence': probability,
        'probability_phishing': probability,
        'probability_legitimate': 1 - probability,
        'message': "⚠️ ALERTE: Email suspect détecté!" if is_phishing else "✅ Email légitime",
        'features': {
            'url_count': features['url_count'],
            'suspicious_words': features['suspicious_word_count'],
            'has_shortened_url': bool(features['has_shortened_url']),
            'exclamation_marks': features['exclamation_count'],
            'urgency_score': features['urgency_score'],
            'threat_score': features['threat_score']
        },
        'mode': 'règles simples'
    }

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 DÉMARRAGE DU SERVEUR")
    print("="*50)
    print(f"📊 Modèle chargé: {'OUI' if model else 'NON (mode règles simples)'}")
    print(f"🌐 Serveur: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)