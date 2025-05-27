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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_auc_score)
import pickle

# NLTK for stop words
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}

class SMSSpamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def download_data(self):
        """Download the SMS spam dataset"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        if not os.path.exists('data/smsspamcollection.zip'):
            print("Downloading SMS Spam Collection dataset...")
            urllib.request.urlretrieve(url, 'data/smsspamcollection.zip')
            print("Dataset downloaded successfully!")
        
        # Extract the zip file
        with zipfile.ZipFile('data/smsspamcollection.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')
            
    def load_data(self):
        """Load and preprocess the SMS spam dataset"""
        # Download data if not exists
        self.download_data()
        
        # Read the TSV file
        df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
        
        print(f"Dataset loaded: {len(df)} messages")
        print(f"Spam messages: {sum(df['label'] == 'spam')} ({sum(df['label'] == 'spam')/len(df)*100:.1f}%)")
        print(f"Ham messages: {sum(df['label'] == 'ham')} ({sum(df['label'] == 'ham')/len(df)*100:.1f}%)")
        
        return df
    
    def preprocess_text(self, text):
        """Clean and preprocess text messages"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # Remove punctuation and digits
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove stop words
        words = text.split()
        words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        
        return ' '.join(words)
    
    def exploratory_analysis(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nDataset Info:")
        print(df.info())
        
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        
        # Message length analysis
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        
        print(f"\nAverage message length:")
        print(f"Spam: {df[df['label']=='spam']['message_length'].mean():.1f} characters")
        print(f"Ham: {df[df['label']=='ham']['message_length'].mean():.1f} characters")
        
        print(f"\nAverage word count:")
        print(f"Spam: {df[df['label']=='spam']['word_count'].mean():.1f} words")
        print(f"Ham: {df[df['label']=='ham']['word_count'].mean():.1f} words")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Label distribution
        df['label'].value_counts().plot(kind='bar', ax=axes[0,0], color=['green', 'red'])
        axes[0,0].set_title('Label Distribution')
        axes[0,0].set_ylabel('Count')
        
        # Message length distribution
        df.boxplot(column='message_length', by='label', ax=axes[0,1])
        axes[0,1].set_title('Message Length by Label')
        
        # Word count distribution
        df.boxplot(column='word_count', by='label', ax=axes[1,0])
        axes[1,0].set_title('Word Count by Label')
        
        # Most common words in spam
        spam_messages = df[df['label'] == 'spam']['message']
        all_spam_words = ' '.join(spam_messages).lower().split()
        spam_word_freq = Counter(all_spam_words)
        
        common_spam_words = dict(spam_word_freq.most_common(10))
        axes[1,1].bar(common_spam_words.keys(), common_spam_words.values(), color='red', alpha=0.7)
        axes[1,1].set_title('Top 10 Words in Spam Messages')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def prepare_features(self, df):
        """Prepare features using TF-IDF vectorization"""
        print("\n" + "="*50)
        print("FEATURE PREPARATION")
        print("="*50)
        
        # Preprocess messages
        print("Preprocessing messages...")
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Sample of preprocessing
        print("\nSample of text preprocessing:")
        for i in range(3):
            print(f"Original: {df.iloc[i]['message']}")
            print(f"Processed: {df.iloc[i]['processed_message']}\n")
        
        # Split data
        X = df['processed_message']
        y = df['label'].map({'ham': 0, 'spam': 1})
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} messages")
        print(f"Test set: {len(self.X_test)} messages")
        
        # TF-IDF Vectorization
        print("\nApplying TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            max_df=0.95,
            min_df=2,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_models(self, X_train_tfidf, X_test_tfidf):
        """Train and evaluate multiple models"""
        print("\n" + "="*50)
        print("MODEL TRAINING & EVALUATION")
        print("="*50)
        
        models = {
            'Multinomial Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Fit model
            model.fit(X_train_tfidf, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test_tfidf)
            y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train_tfidf, X_test_tfidf):
        """Perform hyperparameter tuning with Grid Search"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grids
        param_grids = {
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1.0, 10.0, 100.0], 'penalty': ['l1', 'l2']}
            }
        }
        
        best_results = {}
        
        for name, config in param_grids.items():
            print(f"\nTuning {name}...")
            
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_tfidf, self.y_train)
            
            # Best model predictions
            y_pred = grid_search.best_estimator_.predict(X_test_tfidf)
            y_pred_proba = grid_search.best_estimator_.predict_proba(X_test_tfidf)[:, 1]
            
            best_results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
            print(f"Test F1-score: {f1_score(self.y_test, y_pred):.4f}")
        
        return best_results
    
    def visualize_results(self, results):
        """Create visualizations for model evaluation"""
        print("\n" + "="*50)
        print("VISUALIZATION")
        print("="*50)
        
        # Select best model based on F1-score
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        self.best_model = results[best_model_name]['model']
        
        print(f"Best model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[name][metric] for name in model_names]
            axes[0,0].bar(x + i*width, values, width, label=metric.capitalize())
        
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Comparison')
        axes[0,0].set_xticks(x + width * 1.5)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        
        # 2. Confusion Matrix for best model
        best_predictions = results[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, best_predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # 3. Precision-Recall Curve
        best_probabilities = results[best_model_name]['probabilities']
        precision, recall, _ = precision_recall_curve(self.y_test, best_probabilities)
        
        axes[1,0].plot(recall, precision, marker='.', label=best_model_name)
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        
        # 4. Feature importance (if available)
        if hasattr(self.best_model, 'feature_log_prob_'):
            # For Naive Bayes, show most discriminative features
            feature_names = self.vectorizer.get_feature_names_out()
            log_prob_diff = (self.best_model.feature_log_prob_[1] - 
                           self.best_model.feature_log_prob_[0])
            
            # Top spam indicators
            top_spam_indices = log_prob_diff.argsort()[-10:]
            top_spam_features = [feature_names[i] for i in top_spam_indices]
            top_spam_scores = log_prob_diff[top_spam_indices]
            
            axes[1,1].barh(range(len(top_spam_features)), top_spam_scores, color='red', alpha=0.7)
            axes[1,1].set_yticks(range(len(top_spam_features)))
            axes[1,1].set_yticklabels(top_spam_features)
            axes[1,1].set_title('Top Spam Indicators')
            axes[1,1].set_xlabel('Log Probability Difference')
        
        elif hasattr(self.best_model, 'coef_'):
            # For Logistic Regression, show coefficients
            feature_names = self.vectorizer.get_feature_names_out()
            coefficients = self.best_model.coef_[0]
            
            # Top spam indicators
            top_indices = coefficients.argsort()[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_coefs = coefficients[top_indices]
            
            axes[1,1].barh(range(len(top_features)), top_coefs, color='red', alpha=0.7)
            axes[1,1].set_yticks(range(len(top_features)))
            axes[1,1].set_yticklabels(top_features)
            axes[1,1].set_title('Top Spam Indicators (Coefficients)')
            axes[1,1].set_xlabel('Coefficient Value')
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name
    
    def error_analysis(self, results, best_model_name):
        """Analyze model errors"""
        print("\n" + "="*50)
        print("ERROR ANALYSIS")
        print("="*50)
        
        best_predictions = results[best_model_name]['predictions']
        
        # Find false positives and false negatives
        false_positives = (self.y_test == 0) & (best_predictions == 1)
        false_negatives = (self.y_test == 1) & (best_predictions == 0)
        
        print(f"False Positives: {sum(false_positives)}")
        print(f"False Negatives: {sum(false_negatives)}")
        
        # Show examples
        test_messages = self.X_test.reset_index(drop=True)
        
        print("\nFalse Positive Examples (Ham predicted as Spam):")
        fp_indices = np.where(false_positives)[0]
        for i, idx in enumerate(fp_indices[:3]):
            print(f"{i+1}. {test_messages.iloc[idx]}")
        
        print("\nFalse Negative Examples (Spam predicted as Ham):")
        fn_indices = np.where(false_negatives)[0]
        for i, idx in enumerate(fn_indices[:3]):
            print(f"{i+1}. {test_messages.iloc[idx]}")
    
    def save_model(self, best_model_name=None):
        """Save the trained model and vectorizer"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.best_model,
            'model_name': best_model_name
        }
        
        with open('models/spam_detector.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved as 'models/spam_detector.pkl'")
    
    def predict_message(self, message):
        """Predict if a single message is spam or ham"""
        if self.vectorizer is None or self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess the message
        processed_message = self.preprocess_text(message)
        
        # Vectorize
        message_tfidf = self.vectorizer.transform([processed_message])
        
        # Predict
        prediction = self.best_model.predict(message_tfidf)[0]
        probability = self.best_model.predict_proba(message_tfidf)[0]
        
        result = "🔴 SPAM" if prediction == 1 else "🟢 HAM"
        confidence = probability[prediction]
        
        return result, confidence
    
    def interactive_demo(self):
        """Interactive command line demo"""
        print("\n" + "="*50)
        print("INTERACTIVE SMS SPAM DETECTOR")
        print("="*50)
        print("Type 'quit' to exit")
        
        while True:
            message = input("\nEnter SMS message: ").strip()
            
            if message.lower() == 'quit':
                break
            
            if message:
                try:
                    result, confidence = self.predict_message(message)
                    print(f"Prediction: {result} (Confidence: {confidence:.3f})")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Please enter a message!")

def main():
    """Main function to run the complete SMS spam detection pipeline"""
    detector = SMSSpamDetector()
    
    try:
        # Load and explore data
        df = detector.load_data()
        df = detector.exploratory_analysis(df)
        
        # Prepare features
        X_train_tfidf, X_test_tfidf = detector.prepare_features(df)
        
        # Train baseline models
        results = detector.train_models(X_train_tfidf, X_test_tfidf)
        
        # Hyperparameter tuning
        tuned_results = detector.hyperparameter_tuning(X_train_tfidf, X_test_tfidf)
        
        # Combine results
        all_results = {**results, **{f"Tuned {k}": v for k, v in tuned_results.items()}}
        
        # Visualize results
        best_model_name = detector.visualize_results(all_results)
        
        # Error analysis
        detector.error_analysis(all_results, best_model_name)
        
        # Save model
        detector.save_model(best_model_name)
        
        # Interactive demo
        detector.interactive_demo()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
