import streamlit as st
import pickle
import os
import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Enhanced SMS Spam Detector",
    page_icon="📱",
    layout="wide"
)

@st.cache_data
def load_model():
    """Load the enhanced trained model"""
    try:
        with open('models/enhanced_spam_detector.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Enhanced model not found! Please run 'python improved_main.py' first to train the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features(text):
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

def create_feature_dataframe(texts, model_data):
    """Convert extracted features to DataFrame with proper numeric types"""
    if isinstance(texts, str):
        texts = [texts]
        
    feature_list = []
    for text in texts:
        features = extract_features(text)
        feature_list.append(features)
    
    # Create DataFrame and ensure all columns are numeric
    df = pd.DataFrame(feature_list).fillna(0)
    
    # Get expected feature names from model if available
    if model_data and 'feature_names' in model_data:
        expected_features = model_data['feature_names']
        
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

def enhanced_preprocess_text(text):
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

def get_spam_reasons(text, features):
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

def predict_message(message, model_data):
    """Predict if message is spam or ham with enhanced features"""
    try:
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        
        # Preprocess text
        processed_text = enhanced_preprocess_text(message)
        
        # Get text features
        text_features = vectorizer.transform([processed_text])
        
        # Get engineered features
        feature_df = create_feature_dataframe([message], model_data)
        feature_sparse = csr_matrix(feature_df.values.astype(np.float64))
        
        # Combine features
        combined_features = hstack([text_features, feature_sparse])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0]
        
        # Get spam reasons
        reasons = get_spam_reasons(message, feature_df.iloc[0])
        
        return prediction, probabilities, reasons
        
    except Exception as e:
        st.error(f"Error processing message: {e}")
        return None, None, []

def main():
    st.title("📱 Enhanced SMS Spam Detector")
    st.markdown("**Advanced AI-powered spam detection with feature analysis**")
    st.markdown("---")
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    # Sidebar with model info
    st.sidebar.title("🤖 Model Information")
    st.sidebar.success(f"**Model:** {model_data.get('model_name', 'Unknown')}")
    st.sidebar.info("**Features:** TF-IDF + 25+ engineered features")
    st.sidebar.info("**Techniques:** URL analysis, keyword detection, pattern recognition")
    
    if 'results' in model_data:
        st.sidebar.markdown("### 📊 Performance Metrics")
        for model_name, metrics in model_data['results'].items():
            if model_name == model_data.get('model_name'):
                st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                st.sidebar.metric("F1-Score", f"{metrics.get('f1', 0):.1%}")
                break
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Enter SMS Message")
        
        # Check for session state message (from examples)
        default_message = ""
        if 'example_message' in st.session_state:
            default_message = st.session_state['example_message']
            del st.session_state['example_message']
        
        message = st.text_area(
            "Type or paste your SMS message here:",
            value=default_message,
            height=150,
            placeholder="Example: Alert: Unusual login attempt detected. Reset your password now at http://secure-login-reset.net"
        )
        
        analyze_btn = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
        
        if analyze_btn or default_message:
            if message.strip():
                with st.spinner("Analyzing message..."):
                    prediction, probabilities, reasons = predict_message(message, model_data)
                
                if prediction is not None:
                    # Display result
                    col_result, col_confidence = st.columns([1, 1])
                    
                    with col_result:
                        if prediction == 1:  # Spam
                            st.error("🔴 **SPAM DETECTED**")
                            if reasons:
                                st.markdown(f"**Reasons:** {', '.join(reasons)}")
                        else:  # Ham
                            st.success("🟢 **LEGITIMATE MESSAGE**")
                    
                    with col_confidence:
                        confidence = max(probabilities)
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Probability breakdown
                    st.subheader("📊 Probability Breakdown")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.metric(
                            "🟢 Ham Probability", 
                            f"{probabilities[0]:.1%}",
                            delta=f"{probabilities[0] - probabilities[1]:.1%}" if probabilities[0] > probabilities[1] else None
                        )
                    
                    with prob_col2:
                        st.metric(
                            "🔴 Spam Probability", 
                            f"{probabilities[1]:.1%}",
                            delta=f"{probabilities[1] - probabilities[0]:.1%}" if probabilities[1] > probabilities[0] else None
                        )
                    
                    # Confidence visualization
                    st.progress(confidence)
                    
                    # Feature analysis
                    if prediction == 1 and reasons:
                        with st.expander("🔍 Detailed Analysis"):
                            feature_df = create_feature_dataframe([message], model_data)
                            features = feature_df.iloc[0]
                            
                            if features['has_url'] > 0:
                                st.warning(f"⚠️ Contains {int(features['url_count'])} URL(s)")
                            if features['has_phone'] > 0:
                                st.warning(f"📞 Contains {int(features['phone_count'])} phone number(s)")
                            if features['capital_ratio'] > 0.2:
                                st.warning(f"📢 High capital letter ratio: {features['capital_ratio']:.1%}")
                            if features['exclamation_count'] > 1:
                                st.warning(f"❗ Multiple exclamation marks: {int(features['exclamation_count'])}")
            else:
                st.warning("⚠️ Please enter a message to analyze!")
    
    with col2:
        st.subheader("📝 Test Examples")
        
        # Modern spam examples
        st.markdown("**🔴 Modern Spam Examples:**")
        spam_examples = [
            ("Delivery Alert", "Your parcel is being held by customs. Pay the import fee immediately here: www.delivery-fee-pay.com/12345"),
            ("Fake Prize", "Congratulations! You've won a $500 Amazon gift card. Click https://free-gift-claim.xyz and enter your details"),
            ("Bank Phishing", "URGENT: Your bank account has been temporarily locked. Verify your identity now at http://secure-bank-verify.com"),
            ("IRS Scam", "This is the IRS. You owe back taxes. Pay immediately at https://irs-payments.org to avoid penalties."),
            ("Mobile Renewal", "Your mobile service has expired. Renew within 24 hrs: renew-service-mobile.com")
        ]
        
        for i, (title, example) in enumerate(spam_examples):
            if st.button(f"🔴 {title}", key=f"spam_{i}", use_container_width=True):
                st.session_state['example_message'] = example
                st.rerun()
        
        st.markdown("**🟢 Legitimate Examples:**")
        ham_examples = [
            ("Casual Plans", "Lunch at the new café tomorrow at 12:30? Let me know if that works for you."),
            ("Family Message", "Mom's birthday dinner is next Friday at 7 PM. Can you bring the salad?"),
            ("Appointment", "Your appointment with Dr. Smith is confirmed for June 3 at 9:00 AM"),
            ("Work Message", "Meeting rescheduled to 3 PM tomorrow. Conference room B."),
            ("Friend Message", "Just saw your message—I'll call you back in about an hour")
        ]
        
        for i, (title, example) in enumerate(ham_examples):
            if st.button(f"🟢 {title}", key=f"ham_{i}", use_container_width=True):
                st.session_state['example_message'] = example
                st.rerun()
    
    # Advanced features section
    st.markdown("---")
    st.subheader("🛡️ Advanced Detection Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("🔗 URL Analysis"):
            st.markdown("""
            - **Suspicious domains** (.xyz, .click, .tk)
            - **Short URL detection** (bit.ly, tinyurl)
            - **Phishing URL patterns**
            - **Domain reputation scoring**
            """)
    
    with col2:
        with st.expander("🎯 Keyword Detection"):
            st.markdown("""
            - **Urgency keywords** (urgent, immediate, act now)
            - **Money-related terms** (free, win, prize, cash)
            - **Authority impersonation** (bank, IRS, Apple)
            - **Delivery scam patterns** (parcel, customs, fee)
            """)
    
    with col3:
        with st.expander("📊 Pattern Analysis"):
            st.markdown("""
            - **Text characteristics** (length, capitals)
            - **Punctuation patterns** (multiple !!!)
            - **Phone number detection**
            - **Emoji and special characters**
            """)
    
    # Common spam indicators
    st.markdown("---")
    st.subheader("⚠️ Common Spam Indicators")
    
    with st.expander("How to Identify Spam Messages"):
        st.markdown("""
        ### 🚨 High-Risk Indicators:
        - **Urgent calls to action** → "Act now!", "Limited time!", "Expires soon!"
        - **Fake authority impersonation** → Claims to be from banks, government, tech companies
        - **Suspicious links** → Short URLs, unfamiliar domains, misspelled company names
        - **Requests for personal info** → Passwords, PINs, account details, verification
        - **Too-good-to-be-true offers** → "You've won!", "Free money!", "Guaranteed approval!"
        
        ### 📱 Modern Scam Types:
        - **Delivery scams** → Fake customs fees, package held, shipping problems
        - **Account verification** → "Confirm your account", "Verify identity"  
        - **Prize/lottery scams** → Congratulations messages, gift cards, cash prizes
        - **Technical support** → "Your computer is infected", "Update required"
        - **Romance/dating scams** → Unexpected romantic messages from strangers
        
        ### ✅ Legitimate Message Signs:
        - **Personal context** → References to real conversations, people, or events
        - **Proper grammar** → Well-written without excessive punctuation
        - **Expected contacts** → From known numbers or verified businesses
        - **No urgent requests** → No pressure to act immediately
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 20px;'>"
        "🔒 <strong>Enhanced SMS Spam Detector</strong> • "
        "Built with Streamlit & Machine Learning • "
        "Advanced Feature Engineering for Modern Threats"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
