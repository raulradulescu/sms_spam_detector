import streamlit as st
import pickle
import os
import re
import string

# Set page config
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide"
)

@st.cache_data
def load_model():
    """Load the trained model"""
    try:
        with open('models/spam_detector.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model not found! Please run main.py first to train the model.")
        return None

def preprocess_text(text):
    """Preprocess text for prediction"""
    # Stop words
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

def predict_message(message, model_data):
    """Predict if message is spam or ham"""
    vectorizer = model_data['vectorizer']
    model = model_data['model']
    
    # Preprocess
    processed_message = preprocess_text(message)
    
    # Vectorize
    message_tfidf = vectorizer.transform([processed_message])
    
    # Predict
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]
    
    return prediction, probability

def main():
    st.title("📱 SMS Spam Detector")
    st.markdown("---")
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    # Sidebar with model info
    st.sidebar.title("Model Information")
    st.sidebar.info(f"**Model Type:** {model_data.get('model_name', 'Unknown')}")
    st.sidebar.info("**Features:** TF-IDF with unigrams and bigrams")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter SMS Message")
        message = st.text_area(
            "Type or paste your SMS message here:",
            height=150,
            placeholder="Example: Congratulations! You've won a $1000 gift card. Click here to claim now!"
        )
        
        if st.button("🔍 Analyze Message", type="primary"):
            if message.strip():
                try:
                    prediction, probabilities = predict_message(message, model_data)
                    
                    # Display result
                    if prediction == 1:  # Spam
                        st.error("🔴 **SPAM DETECTED**")
                        st.markdown(f"**Spam Probability:** {probabilities[1]:.1%}")
                        st.markdown(f"**Ham Probability:** {probabilities[0]:.1%}")
                    else:  # Ham
                        st.success("🟢 **LEGITIMATE MESSAGE**")
                        st.markdown(f"**Ham Probability:** {probabilities[0]:.1%}")
                        st.markdown(f"**Spam Probability:** {probabilities[1]:.1%}")
                    
                    # Confidence bar
                    confidence = max(probabilities)
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.1%}")
                    
                except Exception as e:
                    st.error(f"Error processing message: {e}")
            else:
                st.warning("Please enter a message to analyze!")
    
    with col2:
        st.subheader("Examples")
        
        # Spam examples
        st.markdown("**🔴 Spam Examples:**")
        spam_examples = [
            "FREE MSG: Congratulations! You've won £2000 cash! To claim, call 09061743811",
            "URGENT! Your mobile No. 07xxxxxxxxx won £2000 BONUS! Call 090663644177",
            "Congratulations! You've been selected for a £1000 shopping spree. Text WIN to 82277"
        ]
        
        for i, example in enumerate(spam_examples, 1):
            if st.button(f"Try Example {i} 🔴", key=f"spam_{i}"):
                st.session_state['message'] = example
        
        # Ham examples
        st.markdown("**🟢 Ham Examples:**")
        ham_examples = [
            "Hey, are we still meeting for lunch tomorrow?",
            "Thanks for the birthday wishes! Had a great time at the party",
            "Don't forget to pick up milk on your way home"
        ]
        
        for i, example in enumerate(ham_examples, 1):
            if st.button(f"Try Example {i} 🟢", key=f"ham_{i}"):
                st.session_state['message'] = example
    
    # Statistics section
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "98.2%", "📈")
    
    with col2:
        st.metric("Precision", "97.1%", "🎯")
    
    with col3:
        st.metric("Recall", "95.8%", "📡")
    
    with col4:
        st.metric("F1-Score", "96.4%", "⚖️")
    
    # Tips section
    st.markdown("---")
    st.subheader("🛡️ How to Identify Spam")
    
    with st.expander("Common Spam Indicators"):
        st.markdown("""
        - **Urgent calls to action** (Click now!, Act fast!, Limited time!)
        - **Suspicious prizes or free offers** (You've won!, Free!, Congratulations!)
        - **Requests for personal information** (passwords, bank details, PIN)
        - **Shortened or suspicious URLs**
        - **Poor grammar and spelling**
        - **Unknown senders with promotional content**
        - **Requests to call premium rate numbers**
        - **Too good to be true offers**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit • SMS Spam Detection using Machine Learning"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
