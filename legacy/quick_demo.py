import pickle
import re
import string

def load_model():
    """Load the saved model"""
    with open('models/spam_detector.pkl', 'rb') as f:
        return pickle.load(f)
    
def preprocess_text(text):
    """Simple preprocessing"""
    STOP_WORDS = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
    
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

def predict_spam(message, model_data):
    """Predict if message is spam"""
    vectorizer = model_data['vectorizer']
    model = model_data['model']
    
    # Preprocess
    processed = preprocess_text(message)
    
    # Vectorize
    message_tfidf = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]
    
    return prediction, probability

def main():
    """Simple CLI demo"""
    print("📱 SMS Spam Detector - Quick Demo")
    print("=" * 40)
    
    try:
        # Load model
        model_data = load_model()
        print("✅ Model loaded successfully!")
        print(f"Model: {model_data.get('model_name', 'Unknown')}")
        print("-" * 40)
        
        # Test examples
        test_messages = [
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT! You've won £1000! Call now: 09061743811",
            "Can you pick up some milk on your way home?",
            "FREE MSG: Click here to claim your prize!",
            "Meeting rescheduled to 3 PM. See you there!"
        ]
        
        print("Testing example messages:\n")
        
        for i, message in enumerate(test_messages, 1):
            prediction, probabilities = predict_spam(message, model_data)
            
            result = "🔴 SPAM" if prediction == 1 else "🟢 HAM"
            confidence = max(probabilities)
            
            print(f"{i}. Message: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            print(f"   Result: {result} (Confidence: {confidence:.1%})")
            print()
        
        # Interactive mode
        print("-" * 40)
        print("Interactive Mode (type 'quit' to exit):")
        
        while True:
            try:
                message = input("\nEnter SMS: ").strip()
                
                if message.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if message:
                    prediction, probabilities = predict_spam(message, model_data)
                    result = "🔴 SPAM" if prediction == 1 else "🟢 HAM"
                    confidence = max(probabilities)
                    
                    print(f"Result: {result} (Confidence: {confidence:.1%})")
                    
                    if prediction == 1:
                        print(f"Spam probability: {probabilities[1]:.1%}")
                    else:
                        print(f"Ham probability: {probabilities[0]:.1%}")
                else:
                    print("⚠️ Please enter a message!")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("Please run 'python main.py' first to train the model.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    main()
