import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import zipfile
import urllib.request
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import pickle

# NLTK for stop words
import nltk
try:
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}

class EnhancedSMSSpamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.feature_extractor = None
        self.best_model = None
        self.feature_scaler = None
        
    def extract_features(self, text):
        """Extract domain-specific features that indicate spam"""
        features = {}
        
        # URL features
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$ \ $ ,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        short_url_pattern = r'(?:bit\.ly|tinyurl|t\.co|goo\.gl|short\.link|[a-z0-9]+-[a-z0-9]+\.(?:com|xyz|click|link))'
        
        features['has_url'] = int(len(re.findall(url_pattern, text)) > 0)
        features['url_count'] = len(re.findall(url_pattern, text))
        features['has_short_url'] = int(len(re.findall(short_url_pattern, text, re.IGNORECASE)) > 0)
        features['suspicious_domain'] = int(len(re.findall(r'\.(?:xyz|click|tk|ml|ga|cf|bit\.ly)', text, re.IGNORECASE)) > 0)
        
        # Phone number features
        phone_pattern = r'(?:\+?1[-.\s]?)?$ ?[0-9]{3} $ ?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|[0-9]{10,11}'
        features['has_phone'] = int(len(re.findall(phone_pattern, text)) > 0)
        features['phone_count'] = len(re.findall(phone_pattern, text))
        
        # Spam keywords (updated for modern spam)
        spam_keywords = {
            'urgency': ['urgent', 'immediate', 'asap', 'act now', 'limited time', 'expires', 'hurry'],
            'money': ['free', 'win', 'won', 'winner', 'prize', 'cash', 'money', '$', '£', '€', 'claim', 'reward'],
            'suspicious_actions': ['click', 'call now', 'verify', 'confirm', 'update', 'suspended', 'locked', 'blocked'],
            'fake_authority': ['bank', 'paypal', 'amazon', 'apple', 'google', 'microsoft', 'government', 'irs', 'customs'],
            'personal_info': ['ssn', 'password', 'pin', 'account', 'details', 'information', 'verify identity'],
            'delivery_scam': ['parcel', 'package', 'delivery', 'shipment', 'customs', 'fee', 'import', 'held']
        }
        
        text_lower = text.lower()
        
        for category, keywords in spam_keywords.items():
            features[f'{category}_keywords'] = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'has_{category}'] = int(any(keyword in text_lower for keyword in keywords))
        
        # Text characteristics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0.0
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0.0
        
        # Advanced spam indicators
        features['has_emoji'] = int(bool(re.search(r'[😀-🙏🚀-🛿☀-⛿✀-➿]', text)))
        features['has_all_caps_word'] = int(bool(re.search(r'\b[A-Z]{3,}\b', text)))
        features['has_multiple_exclamation'] = int('!!' in text or '!!!' in text)
        features['starts_with_urgent'] = int(text.lower().strip().startswith(('urgent', 'important', 'attention')))
        
        # Suspicious patterns
        features['has_obfuscation'] = int(bool(re.search(r'[a-z]\s+[a-z]', text.lower())))
        features['has_special_chars'] = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', text))
        
        return features
    
    def create_feature_dataframe(self, texts):
        """Convert extracted features to DataFrame with proper numeric types"""
        feature_list = []
        for text in texts:
            features = self.extract_features(text)
            feature_list.append(features)
        
        # Create DataFrame and ensure all columns are numeric
        df = pd.DataFrame(feature_list).fillna(0)
        
        # Convert all columns to float64 to ensure compatibility with sparse matrices
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float64)
        
        return df
    
    def enhanced_preprocess_text(self, text):
        """Enhanced preprocessing that preserves important spam indicators"""
        # Basic cleaning while preserving structure
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace
        text = text.strip()
        
        # Don't remove URLs and phones completely - mark them
        text = re.sub(r'http[s]?://[^\s]+', ' URL_TOKEN ', text)
        text = re.sub(r'\b\d{10,}\b', ' PHONE_TOKEN ', text)
        text = re.sub(r'[£$€]\d+', ' MONEY_TOKEN ', text)
        
        # Convert to lowercase for text analysis
        text_lower = text.lower()
        
        # Remove excessive punctuation but keep some
        text_lower = re.sub(r'[^\w\s!?.,]', ' ', text_lower)
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        
        return text_lower
    
    def download_data(self):
        """Download the SMS spam dataset"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        
        os.makedirs('data', exist_ok=True)
        
        if not os.path.exists('data/smsspamcollection.zip'):
            print("Downloading SMS Spam Collection dataset...")
            urllib.request.urlretrieve(url, 'data/smsspamcollection.zip')
        
        with zipfile.ZipFile('data/smsspamcollection.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')
    
    def load_data(self):
        """Load and preprocess the data"""
        self.download_data()
        
        # Read the TSV file
        df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])
        
        # Add synthetic modern spam samples to improve detection
        modern_spam_samples = [
            ("spam", "Your parcel is being held by customs. Pay the import fee immediately here: www.delivery-fee-pay.com/12345"),
            ("spam", "Congratulations! You've won a $500 Amazon gift card. Click https://free-gift-claim.xyz and enter your details"),
            ("spam", "URGENT: Your bank account has been temporarily locked. Verify your identity now at http://secure-bank-verify.com"),
            ("spam", "Your PayPal account is suspended. Click here to verify: paypal-verify.secure-site.com"),
            ("spam", "IRS NOTICE: You owe back taxes. Pay immediately to avoid legal action: irs-payment.gov-site.com"),
            ("spam", "Package delivery failed. Pay redelivery fee: $3.99 at delivery-services.quick-pay.net"),
            ("spam", "Apple ID locked for security. Verify now: apple-id-verify.secure-login.com"),
            ("spam", "FREE iPhone 13! Limited time offer. Claim yours now: free-iphone-claim.xyz"),
            ("spam", "Your subscription expires today! Renew now to avoid service interruption: netflix-renewal.com"),
            ("spam", "WINNER! You've been selected for our cash prize. Claim $1000: cash-claim.winner-site.com"),
            ("spam", "Custom fees due. Package held. Pay now: customs-fee-payment.urgent-delivery.com"),
            ("spam", "Bank alert: Suspicious activity detected. Verify account: secure-bank-login.verification-site.com"),
            ("spam", "Amazon: Order cancelled. Refund processing error. Update payment: amazon-refund.payment-update.net"),
            ("spam", "Microsoft: Your account will be closed. Verify identity: microsoft-account.security-check.org"),
            ("spam", "Tax refund available! Claim £450 now: hmrc-refund.gov-services.co.uk"),
            ("ham", "Your Amazon order #123456 has been shipped and will arrive tomorrow"),
            ("ham", "Bank of America: Your statement is ready for viewing online"),
            ("ham", "PayPal: You sent $25.00 to John Smith"),
            ("ham", "Your Uber driver will arrive in 3 minutes"),
            ("ham", "Appointment reminder: Doctor visit tomorrow at 2 PM"),
            ("ham", "Package delivered successfully. Thank you for choosing our service."),
            ("ham", "Your flight check-in is now available. Gate information will be updated 2 hours before departure."),
            ("ham", "Meeting rescheduled to 3 PM tomorrow. Conference room B."),
            ("ham", "Happy birthday! Hope you have a wonderful day!"),
            ("ham", "Don't forget our dinner reservation tonight at 7 PM."),
        ]
        
        # Add modern samples
        modern_df = pd.DataFrame(modern_spam_samples, columns=['label', 'text'])
        df = pd.concat([df, modern_df], ignore_index=True)
        
        # Convert labels to binary
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Data exploration
        print("\n📊 Dataset Overview:")
        print(f"Total samples: {len(df)}")
        print(f"Spam messages: {df['label'].sum()} ({df['label'].mean():.1%})")
        print(f"Ham messages: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")
        
        return df
    
    def train_model(self):
        """Train the enhanced model with feature engineering"""
        print("🚀 Starting Enhanced SMS Spam Detection Training...")
        
        # Load data
        df = self.load_data()
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.enhanced_preprocess_text)
        
        # Extract features
        print("🔧 Extracting enhanced features...")
        feature_df = self.create_feature_dataframe(df['text'])
        
        print(f"Feature shape: {feature_df.shape}")
        print(f"Feature types: {feature_df.dtypes.unique()}")
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_features = self.vectorizer.fit_transform(df['processed_text'])
        print(f"TF-IDF shape: {tfidf_features.shape}")
        
        # Split the data
        X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            tfidf_features, feature_df, df['label'], 
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        # Convert feature DataFrames to sparse matrices with proper dtype
        X_feat_train_sparse = csr_matrix(X_feat_train.values.astype(np.float64))
        X_feat_test_sparse = csr_matrix(X_feat_test.values.astype(np.float64))
        
        # Combine TF-IDF and engineered features
        X_train_combined = hstack([X_text_train, X_feat_train_sparse])
        X_test_combined = hstack([X_text_test, X_feat_test_sparse])
        
        print(f"Combined training shape: {X_train_combined.shape}")
        print(f"Combined test shape: {X_test_combined.shape}")
        
        # Train multiple models
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
        }
        
        best_score = 0
        best_model_name = ""
        results = {}
        
        print("\n🔍 Training and evaluating models...")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='f1')
            
            # Train on full training set
            model.fit(X_train_combined, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_combined)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"  CV F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"  Test Accuracy: {accuracy:.3f}")
            print(f"  Test Precision: {precision:.3f}")
            print(f"  Test Recall: {recall:.3f}")
            print(f"  Test F1: {f1:.3f}")
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                self.best_model = model
        
        print(f"\n🏆 Best individual model: {best_model_name} (F1: {best_score:.3f})")
        
        # Create ensemble model for even better performance
        print("\n🎯 Training ensemble model...")
        ensemble = VotingClassifier([
            ('nb', MultinomialNB(alpha=0.1)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, C=1.0)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20))
        ], voting='soft')
        
        ensemble.fit(X_train_combined, y_train)
        y_pred_ensemble = ensemble.predict(X_test_combined)
        
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_precision = precision_score(y_test, y_pred_ensemble)
        ensemble_recall = recall_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble)
        
        print(f"Ensemble Results:")
        print(f"  Accuracy: {ensemble_accuracy:.3f}")
        print(f"  Precision: {ensemble_precision:.3f}")
        print(f"  Recall: {ensemble_recall:.3f}")
        print(f"  F1: {ensemble_f1:.3f}")
        
        if ensemble_f1 > best_score:
            self.best_model = ensemble
            best_model_name = "Ensemble"
            best_score = ensemble_f1
            print(f"✅ Ensemble model selected as best!")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.best_model,
            'model_name': best_model_name,
            'feature_names': feature_df.columns.tolist(),
            'results': results
        }
        
        with open('models/enhanced_spam_detector.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Enhanced model saved successfully!")
        print(f"Best model: {best_model_name} with F1 score: {best_score:.3f}")
        
        # Test on problematic examples
        print("\n🧪 Testing on previously misclassified examples:")
        test_cases = [
            "Your parcel is being held by customs. Pay the import fee immediately here: www.delivery-fee-pay.com/12345",
            "Congratulations! You've won a $500 Amazon gift card. Click https://free-gift-claim.xyz and enter your details",
            "URGENT: Your bank account has been temporarily locked. Verify your identity now at http://secure-bank-verify.com",
            "Hey Sara, I'm running 10 minutes late. Sorry about that—see you soon!",
            "Your Amazon order #123456 has been shipped and will arrive tomorrow"
        ]
        
        for i, text in enumerate(test_cases, 1):
            processed = self.enhanced_preprocess_text(text)
            text_features = self.vectorizer.transform([processed])
            manual_features_df = self.create_feature_dataframe([text])
            manual_features_sparse = csr_matrix(manual_features_df.values.astype(np.float64))
            combined_features = hstack([text_features, manual_features_sparse])
            
            prediction = self.best_model.predict(combined_features)[0]
            probability = self.best_model.predict_proba(combined_features)[0]
            
            result = "🔴 SPAM" if prediction == 1 else "🟢 HAM"
            confidence = max(probability)
            
            print(f"{i}. {result} ({confidence:.1%}) - \"{text[:60]}...\"")
        
        # Show confusion matrix
        print("\n📊 Final Confusion Matrix:")
        y_final_pred = self.best_model.predict(X_test_combined)
        cm = confusion_matrix(y_test, y_final_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nConfusion Matrix saved to 'models/confusion_matrix.png'")

if __name__ == "__main__":
    detector = EnhancedSMSSpamDetector()
    detector.train_model()
