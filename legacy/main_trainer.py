import os
import sys
import pickle
import pandas as pd
import numpy as np
import pefile
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class MalwareDetector:
    def __init__(self, model_path=None):
        """Initialize the malware detector with an optional pre-trained model."""
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded model from {model_path}")
        else:
            self.model = None
            print("No model loaded. Use train() to create a new model.")

    def extract_features(self, file_path):
        """Extract features from a file for malware detection."""
        features = {}
        
        # Basic file properties
        file_stats = os.stat(file_path)
        features['file_size'] = file_stats.st_size
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_data = f.read()
            features['md5'] = hashlib.md5(file_data).hexdigest()
            features['sha1'] = hashlib.sha1(file_data).hexdigest()
        
        try:
            # Try to parse as PE file (Windows executable)
            pe = pefile.PE(file_path)
            
            # PE Header features
            features['number_of_sections'] = len(pe.sections)
            features['timestamp'] = pe.FILE_HEADER.TimeDateStamp
            
            # Import features
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                features['number_of_imports'] = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
                dll_names = [entry.dll.decode('utf-8', 'ignore').lower() for entry in pe.DIRECTORY_ENTRY_IMPORT]
                
                # Check for suspicious imports
                suspicious_dlls = ['wininet.dll', 'urlmon.dll', 'ws2_32.dll', 'crypt32.dll']
                for dll in suspicious_dlls:
                    features[f'has_{dll.replace(".", "_")}'] = int(dll in dll_names)
            else:
                features['number_of_imports'] = 0
                for dll in ['wininet.dll', 'urlmon.dll', 'ws2_32.dll', 'crypt32.dll']:
                    features[f'has_{dll.replace(".", "_")}'] = 0
            
            # Section features
            features['executable_sections'] = 0
            features['writable_sections'] = 0
            features['entropy_avg'] = 0
            
            for section in pe.sections:
                if section.Characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                    features['executable_sections'] += 1
                if section.Characteristics & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                    features['writable_sections'] += 1
                
                # Calculate entropy for the section
                try:
                    entropy = self._calculate_entropy(section.get_data())
                    features['entropy_avg'] += entropy
                except:
                    pass
            
            if len(pe.sections) > 0:
                features['entropy_avg'] /= len(pe.sections)
            
        except:
            # Not a PE file or error parsing
            features['number_of_sections'] = -1
            features['timestamp'] = -1
            features['number_of_imports'] = -1
            features['executable_sections'] = -1
            features['writable_sections'] = -1
            features['entropy_avg'] = -1
            for dll in ['wininet.dll', 'urlmon.dll', 'ws2_32.dll', 'crypt32.dll']:
                features[f'has_{dll.replace(".", "_")}'] = -1
        
        return features

    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        if not data:
            return 0
        
        entropy = 0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
                
        return entropy

    def train(self, malware_dir, benign_dir, model_save_path='malware_model.pkl'):
        """Train the model using files from malware_dir and benign_dir."""
        # Extract features from malware files
        malware_features = []
        for filename in os.listdir(malware_dir):
            file_path = os.path.join(malware_dir, filename)
            if os.path.isfile(file_path):
                try:
                    features = self.extract_features(file_path)
                    features['is_malware'] = 1
                    malware_features.append(features)
                    print(f"Processed malware: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Extract features from benign files
        benign_features = []
        for filename in os.listdir(benign_dir):
            file_path = os.path.join(benign_dir, filename)
            if os.path.isfile(file_path):
                try:
                    features = self.extract_features(file_path)
                    features['is_malware'] = 0
                    benign_features.append(features)
                    print(f"Processed benign file: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Combine and convert to DataFrame
        all_features = malware_features + benign_features
        df = pd.DataFrame(all_features)
        
        # Remove non-numeric columns (like hashes)
        df = df.select_dtypes(include=[np.number])
        
        # Split data
        X = df.drop('is_malware', axis=1)
        y = df['is_malware']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model
        with open(model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_save_path}")
        
        return self.model

    def scan_file(self, file_path):
        """Scan a file and predict if it's malware or benign."""
        if not self.model:
            raise Exception("No model loaded. Train or load a model first.")
        
        # Extract features
        features = self.extract_features(file_path)
        
        # Convert to DataFrame and keep only numeric columns
        df = pd.DataFrame([features])
        df = df.select_dtypes(include=[np.number])
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        result = {
            'file': os.path.basename(file_path),
            'prediction': 'Malware' if prediction == 1 else 'Benign',
            'confidence': probability[1] if prediction == 1 else probability[0],
            'features': features
        }
        
        return result

def main():
    if len(sys.argv) < 2:
        print("Usage modes:")
        print("  1. Training: python malware_detector.py train <malware_dir> <benign_dir> [<model_save_path>]")
        print("  2. Scanning: python malware_detector.py scan <file_path> <model_path>")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'train':
        if len(sys.argv) < 4:
            print("Training usage: python malware_detector.py train <malware_dir> <benign_dir> [<model_save_path>]")
            return
        
        malware_dir = sys.argv[2]
        benign_dir = sys.argv[3]
        model_save_path = sys.argv[4] if len(sys.argv) > 4 else 'malware_model.pkl'
        
        detector = MalwareDetector()
        detector.train(malware_dir, benign_dir, model_save_path)
        
    elif mode == 'scan':
        if len(sys.argv) < 4:
            print("Scanning usage: python malware_detector.py scan <file_path> <model_path>")
            return
        
        file_path = sys.argv[2]
        model_path = sys.argv[3]
        
        detector = MalwareDetector(model_path)
        result = detector.scan_file(file_path)
        
        print(f"\nScan Results for: {result['file']}")
        print(f"Classification: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nKey Features:")
        for feature, value in result['features'].items():
            if feature not in ['md5', 'sha1']:
                print(f"  {feature}: {value}")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, scan")

if __name__ == "__main__":
    main()
