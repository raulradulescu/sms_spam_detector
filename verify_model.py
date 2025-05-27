import pickle
import os

def verify_model():
    """Check if the model file is valid and show its contents"""
    model_path = 'models/enhanced_spam_detector.pkl'
    
    if not os.path.exists(model_path):
        print("❌ Enhanced model file not found!")
        print("Please run: python improved_main.py")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Model file loaded successfully!")
        print("\n📋 Model Information:")
        print(f"Model type: {type(model_data.get('model'))}")
        print(f"Model name: {model_data.get('model_name', 'Unknown')}")
        print(f"Vectorizer: {type(model_data.get('vectorizer'))}")
        
        if 'feature_names' in model_data:
            print(f"Number of features: {len(model_data['feature_names'])}")
            print(f"Feature names: {model_data['feature_names'][:10]}..." if len(model_data['feature_names']) > 10 else str(model_data['feature_names']))
        
        if 'results' in model_data:
            print(f"\n📊 Training Results:")
            for model_name, metrics in model_data['results'].items():
                print(f"  {model_name}: F1={metrics.get('f1', 0):.3f}, Acc={metrics.get('accuracy', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("The model file may be corrupted. Try retraining with: python improved_main.py")
        return False

if __name__ == "__main__":
    verify_model()
