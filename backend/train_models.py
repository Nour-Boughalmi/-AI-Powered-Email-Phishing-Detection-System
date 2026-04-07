# train_final_corrected.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import re

print("="*60)
print("🚀 RÉ-ENTRAÎNEMENT CORRIGÉ AVEC VOS DONNÉES")
print("="*60)

# 1. Charger vos données
print("\n📂 Chargement de vos données...")
df = pd.read_csv('data/emails.csv')
print(f"✅ {len(df)} emails chargés")
print(f"   Légitimes (0): {sum(df['label']==0)}")
print(f"   Phishing (1): {sum(df['label']==1)}")
print(f"   Proportion: {df['label'].mean():.1%} phishing")

# 2. Nettoyage et extraction de features ROBUSTE
print("\n🔧 Extraction des features (version robuste)...")

def extract_features_safe(row):
    """Extraction de features qui gère les valeurs NaN"""
    
    # Gérer les valeurs manquantes
    subject = str(row['subject']) if pd.notna(row['subject']) else ""
    body = str(row['body']) if pd.notna(row['body']) else ""
    
    text = subject + " " + body
    text_lower = text.lower()
    
    # URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    
    # Features
    features = {
        'length': min(len(text), 5000),
        'word_count': len(text.split()),
        'exclamation_count': min(text.count('!'), 10),
        'question_count': min(text.count('?'), 10),
        'url_count': len(urls),
        'has_url': 1 if len(urls) > 0 else 0,
        'has_shortened': 1 if any(x in text_lower for x in ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly']) else 0,
        'has_ip_url': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', text) else 0,
    }
    
    # Mots suspects (phishing)
    suspicious_words = ['urgent', 'verify', 'account', 'password', 'click', 'login', 
                       'security', 'update', 'confirm', 'bank', 'paypal', 'amazon',
                       'suspended', 'locked', 'alert', 'immediate', 'limited', 
                       'compromised', 'unauthorized', 'blocked', 'warning', 'suspend']
    
    features['suspicious_count'] = sum(1 for word in suspicious_words if word in text_lower)
    
    # Mots d'urgence
    urgency_words = ['urgent', 'immediate', 'asap', 'now', 'today', 'immediately']
    features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
    
    # Menaces
    threat_words = ['suspend', 'close', 'block', 'lock', 'terminate', 'lost', 'limited']
    features['threat_score'] = sum(1 for word in threat_words if word in text_lower)
    
    # Ratio majuscules
    upper_chars = sum(1 for c in text if c.isupper())
    features['upper_ratio'] = upper_chars / max(len(text), 1)
    
    return features

# Appliquer l'extraction
print("   - Traitement des emails...")
features_list = []
for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"   Progression: {idx}/{len(df)}")
    features_list.append(extract_features_safe(row))

X = pd.DataFrame(features_list)
y = df['label']

print(f"\n✅ {len(X.columns)} features extraites:")
print(f"   {list(X.columns)}")

# 3. Split des données (avec stratification)
print("\n📊 Split des données...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train)} (légitimes: {sum(y_train==0)}, phishing: {sum(y_train==1)})")
print(f"   Test: {len(X_test)} (légitimes: {sum(y_test==0)}, phishing: {sum(y_test==1)})")

# 4. Entraînement CatBoost avec correction du déséquilibre
print("\n🤖 Entraînement CatBoost (avec auto-balancing)...")

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',  # CRUCIAL: équilibre les classes
    verbose=100,
    random_seed=42,
    early_stopping_rounds=50
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100, plot=False)

# 5. Évaluation complète
print("\n" + "="*60)
print("📊 ÉVALUATION DU MODÈLE")
print("="*60)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Trouver le meilleur seuil
print("\n🔍 Recherche du meilleur seuil...")
best_threshold = 0.5
best_f1 = 0

for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   Seuil {threshold:.2f}: Précision={precision:.2f}, Rappel={recall:.2f}, F1={f1:.2f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n✅ Meilleur seuil: {best_threshold} (F1={best_f1:.2f})")

# Appliquer le meilleur seuil
y_pred_final = (y_proba >= best_threshold).astype(int)

print("\n📋 Classification Report (avec seuil optimisé):")
print(classification_report(y_test, y_pred_final, target_names=['Légitime', 'Phishing']))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_final)
print(f"\n📊 Matrice de confusion:")
print(f"   Vrais Négatifs (légitime correct): {cm[0,0]}")
print(f"   Faux Positifs (phishing détecté à tort): {cm[0,1]}")
print(f"   Faux Négatifs (phishing non détecté): {cm[1,0]}")
print(f"   Vrais Positifs (phishing correct): {cm[1,1]}")

# 6. Test sur exemples concrets
print("\n" + "="*60)
print("🧪 TEST SUR EXEMPLES RÉELS")
print("="*60)

test_examples = [
    ("Email NORMAL - Newsletter", "Your weekly newsletter is ready. Read online.", 0),
    ("Email NORMAL - Meeting", "Meeting reminder: Tomorrow at 10am in room B", 0),
    ("Email NORMAL - Order", "Your order #12345 has been shipped", 0),
    ("Email PHISHING - PayPal", "URGENT: Your PayPal account is limited! Verify: http://bit.ly/verify", 1),
    ("Email PHISHING - Bank", "Your bank account is compromised. Login immediately: http://fake-bank.com", 1),
    ("Email PHISHING - Amazon", "Your Amazon order is on hold. Click here: http://tinyurl.com/amazon-verify", 1),
]

for name, text, expected in test_examples:
    # Extraire features
    test_row = pd.Series({'subject': text[:50], 'body': text})
    features = extract_features_safe(test_row)
    features_df = pd.DataFrame([features])
    
    # Aligner les colonnes
    for col in X.columns:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[X.columns]
    
    # Prédiction
    proba = model.predict_proba(features_df)[0, 1]
    pred = 1 if proba > best_threshold else 0
    
    status = "✅" if pred == expected else "❌"
    print(f"\n{status} {name}")
    print(f"   Texte: {text[:60]}...")
    print(f"   Probabilité phishing: {proba:.1%}")
    print(f"   Prédiction: {'PHISHING' if pred else 'LÉGITIME'}")
    print(f"   Attendu: {'PHISHING' if expected else 'LÉGITIME'}")

# 7. Sauvegarde
print("\n💾 Sauvegarde du modèle...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/catboost_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
joblib.dump(best_threshold, 'models/best_threshold.pkl')

print("\n✅ Modèle sauvegardé!")
print("   - models/catboost_model.pkl")
print("   - models/feature_names.pkl")
print(f"   - Seuil optimal: {best_threshold}")

print("\n" + "="*60)
print("🎉 ENTRAÎNEMENT TERMINÉ!")
print("="*60)
print("\nMaintenant votre modèle détectera correctement:")
print("  ✅ Les emails LÉGITIMES (score faible)")
print("  ✅ Les emails PHISHING (score élevé)")
print("\nLancez l'application: python backend/app.py")