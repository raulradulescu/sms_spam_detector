import pickle
import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# NLTK for stop words
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}

class EnhancedSpamDemo:
    def __init__(self):
        self.model_data = None
        
    def extract_features(self, text):
        """Extract domain-specific features that indicate spam - same as training"""
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
        if isinstance(texts, str):
            texts = [texts]
            
        feature_list = []
        for text in texts:
            features = self.extract_features(text)
            feature_list.append(features)
        
        # Create DataFrame and ensure all columns are numeric
        df = pd.DataFrame(feature_list).fillna(0)
        
        # Get expected feature names from model if available
        if self.model_data and 'feature_names' in self.model_data:
            expected_features = self.model_data['feature_names']
            
            # Add missing columns with default values
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Reorder columns to match training
            df = df[expected_features]
        
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

    def load_model(self):
        """Load the enhanced model with error handling"""
        try:
            with open('models/enhanced_spam_detector.pkl', 'rb') as f:
                self.model_data = pickle.load(f)
            return True
        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            
            # Try to load the basic model as fallback
            try:
                with open('models/spam_detector.pkl', 'rb') as f:
                    basic_model = pickle.load(f)
                    print("⚠️ Loaded basic model as fallback")
                    return False
            except:
                print("❌ No models found. Please run 'python improved_main.py' first.")
                return False

    def predict_spam(self, message):
        """Predict if message is spam with enhanced features"""
        if not self.model_data:
            return None, None, []
            
        try:
            vectorizer = self.model_data['vectorizer']
            model = self.model_data['model']
            
            # Preprocess text
            processed_text = self.enhanced_preprocess_text(message)
            
            # Get text features
            text_features = vectorizer.transform([processed_text])
            
            # Get engineered features
            feature_df = self.create_feature_dataframe([message])
            feature_sparse = csr_matrix(feature_df.values.astype(np.float64))
            
            # Combine features
            combined_features = hstack([text_features, feature_sparse])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            probabilities = model.predict_proba(combined_features)[0]
            
            # Get spam reasons
            reasons = self.get_spam_reasons(message, feature_df.iloc[0])
            
            return prediction, probabilities, reasons
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None, []
    
    def get_spam_reasons(self, text, features):
        """Explain why a message was classified as spam"""
        reasons = []
        
        # Check individual features
        if features.get('has_url', 0) > 0:
            reasons.append("contains URLs")
        if features.get('suspicious_domain', 0) > 0:
            reasons.append("suspicious domain")
        if features.get('has_urgency', 0) > 0:
            reasons.append("urgent language")
        if features.get('has_money', 0) > 0:
            reasons.append("money/prize terms")
        if features.get('has_fake_authority', 0) > 0:
            reasons.append("authority impersonation")
        if features.get('has_delivery_scam', 0) > 0:
            reasons.append("delivery scam pattern")
        if features.get('has_suspicious_actions', 0) > 0:
            reasons.append("suspicious action requests")
        if features.get('has_personal_info', 0) > 0:
            reasons.append("requests personal info")
        if features.get('capital_ratio', 0) > 0.3:
            reasons.append("excessive capitals")
        if features.get('exclamation_count', 0) > 2:
            reasons.append("multiple exclamations")
        if features.get('has_phone', 0) > 0:
            reasons.append("contains phone numbers")
        
        return reasons

def main():
    """Enhanced CLI demo with better error handling"""
    print("📱 Enhanced SMS Spam Detector")
    print("=" * 50)
    
    demo = EnhancedSpamDemo()
    
    # Try to load model
    if not demo.load_model():
        print("\n🔧 Please run the training script first:")
        print("python improved_main.py")
        return
    
    print("✅ Enhanced model loaded successfully!")
    print(f"Model: {demo.model_data.get('model_name', 'Unknown')}")
    print("-" * 50)
    
    # Test on previously problematic messages
    print("Testing previously problematic messages:\n")
    
    test_cases = [
        ("Your parcel is being held by customs. Pay the import fee immediately here: www.delivery-fee-pay.com/12345", True),
        ("Congratulations! You've won a $500 Amazon gift card. Click https://free-gift-claim.xyz and enter your details", True),
        ("URGENT: Your bank account has been temporarily locked. Verify your identity now at http://secure-bank-verify.com", True),
        ("Happy birthday! 🎉 Hope you have an amazing day—let's catch up this weekend.", False),
        ("Hey Sara, I'm running 10 minutes late. Sorry about that—see you soon!", False),
        ("Don't forget to pick up milk and eggs on your way home. Thanks!", False),
        ("Your Amazon order #123456 has been shipped and will arrive tomorrow", False),
        ("Meeting rescheduled to 3 PM tomorrow. Conference room B.", False)
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, (message, is_spam_expected) in enumerate(test_cases, 1):
        prediction, probabilities, reasons = demo.predict_spam(message)
        
        if prediction is not None:
            is_spam = prediction == 1
            confidence = max(probabilities)
            
            # Check if prediction is correct
            if is_spam == is_spam_expected:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            result = "🔴 SPAM" if is_spam else "🟢 HAM"
            print(f"{status} {i}. {result} (Confidence: {confidence:.1%})")
            print(f"   Message: \"{message[:70]}...\"")
            
            if is_spam and reasons:
                print(f"   Reasons: {', '.join(reasons)}")
            
            if not is_spam:
                print(f"   Ham probability: {probabilities[0]:.1%}")
            
            print()
        else:
            print(f"❌ {i}. Error processing message")
            print()
    
    accuracy = correct_predictions / total_predictions
    print(f"📊 Test Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
    print("-" * 50)
    
    # Interactive mode
    print("\n🎯 Interactive Mode (Press Ctrl+C to exit)")
    print("Enter SMS messages to test:")
    print()
    
    while True:
        try:
            message = input("Enter SMS: ").strip()
            
            if message:
                prediction, probabilities, reasons = demo.predict_spam(message)
                
                if prediction is not None:
                    is_spam = prediction == 1
                    confidence = max(probabilities)
                    
                    result = "🔴 SPAM" if is_spam else "🟢 HAM"
                    print(f"Result: {result} (Confidence: {confidence:.1%})")
                    
                    if is_spam and reasons:
                        print(f"Reasons: {', '.join(reasons)}")
                    else:
                        print(f"Ham probability: {probabilities[0]:.1%}")
                        
                else:
                    print("⚠️ Could not process message")
                    
                print()
            else:
                print("⚠️ Please enter a message!")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
