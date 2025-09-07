import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define the base directory path
BASE_DIR = r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek"

# Set page config
st.set_page_config(
    page_title="Enhanced Fraud Detection via Sequence Embeddings",
    page_icon="🔍",
    layout="wide"
)

# Load enhanced models (with caching)
# Load enhanced models (with caching)
@st.cache_resource
def load_enhanced_models():
    try:
        model_dir = os.path.join(BASE_DIR, "models")
        vectorizer = joblib.load(os.path.join(model_dir, "enhanced_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "enhanced_scaler.pkl"))
        classifier = joblib.load(os.path.join(model_dir, "enhanced_fraud_classifier.pkl"))
        feature_names = joblib.load(os.path.join(model_dir, "enhanced_feature_names.pkl"))
        return vectorizer, scaler, classifier, feature_names
    except FileNotFoundError:
        st.error("Enhanced models not found. Please run the enhanced training script first.")
        return None, None, None, None

# Load sample data for similarity analysis
@st.cache_data
def load_sample_data():
    try:
        data_path = os.path.join(BASE_DIR, "data", "synthetic_claims.csv")
        df = pd.read_csv(data_path)
        df['event_sequence'] = df['event_sequence'].apply(eval)
        df['sequence_text'] = df['event_sequence'].apply(lambda x: ' '.join(x))
        return df
    except FileNotFoundError:
        return None

# Define event types for the dropdown
EVENT_TYPES = [
    "First Notice of Loss",
    "Initial Assessment",
    "Damage Documentation",
    "Witness Statement",
    "Police Report Filed",
    "Expert Consultation",
    "Repair Authorization",
    "Parts Ordered",
    "Repair In Progress",
    "Quality Check",
    "Final Assessment",
    "Payment Processing",
    "Claim Closed"
]

def create_advanced_features_from_sequence(sequence, vectorizer, feature_names):
    """Create advanced features from a sequence for prediction"""
    # Convert sequence to text
    sequence_text = ' '.join(sequence)
    
    # Transform using the trained vectorizer
    vectorized_features = vectorizer.transform([sequence_text]).toarray()
    vectorized_df = pd.DataFrame(vectorized_features, columns=vectorizer.get_feature_names_out())
    
    # Create engineered features
    sequence_length = len(sequence)
    
    engineered_features = {
        'sequence_length': sequence_length,
        'count_expert_consultation': sum(1 for e in sequence if e == 'Expert Consultation'),
        'count_witness_statement': sum(1 for e in sequence if e == 'Witness Statement'),
        'count_police_report_filed': sum(1 for e in sequence if e == 'Police Report Filed'),
        'count_damage_documentation': sum(1 for e in sequence if e == 'Damage Documentation'),
        'count_assessment': sum(1 for e in sequence if 'Assessment' in e),
        'has_assessment_before_documentation': any(
            'Assessment' in sequence[i] and 'Documentation' in sequence[i+1] 
            for i in range(len(sequence)-1)
        ) if len(sequence) > 1 else False,
        'payment_before_assessment': any(
            'Payment' in sequence[i] and any('Assessment' in sequence[j] for j in range(i+1, len(sequence)))
            for i in range(len(sequence)-1)
        ) if len(sequence) > 1 else False
    }
    
    # Add engineered features
    for feature, value in engineered_features.items():
        vectorized_df[feature] = value
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in vectorized_df.columns:
            vectorized_df[feature] = 0
    
    # Reorder columns to match training data
    vectorized_df = vectorized_df[feature_names]
    
    return vectorized_df

def analyze_sequence_pattern(sequence):
    """Analyze a sequence for potential fraud indicators"""
    indicators = []
    
    # Check for too short sequences
    if len(sequence) < 4:
        indicators.append("Sequence is very short (less than 4 events)")
    
    # Check for missing critical events
    critical_events = ['Assessment', 'Documentation']
    for event in critical_events:
        if not any(event in e for e in sequence):
            indicators.append(f"Missing critical event: {event}")
    
    # Check for suspicious repeats
    for event in ['Expert Consultation', 'Witness Statement', 'Assessment']:
        count = sum(1 for e in sequence if event in e)
        if count > 2:
            indicators.append(f"Suspicious repetition of: {event} ({count} times)")
    
    # Check for illogical ordering
    payment_index = next((i for i, e in enumerate(sequence) if 'Payment' in e), -1)
    assessment_index = next((i for i, e in enumerate(sequence) if 'Assessment' in e), -1)
    
    if payment_index != -1 and assessment_index != -1 and payment_index < assessment_index:
        indicators.append("Illogical ordering: Payment before Assessment")
    
    # Check for documentation after payment
    documentation_index = next((i for i, e in enumerate(sequence) if 'Documentation' in e), -1)
    if payment_index != -1 and documentation_index != -1 and payment_index < documentation_index:
        indicators.append("Illogical ordering: Payment before Documentation")
    
    return indicators

def main():
    st.title("🔍 Enhanced Fraud Detection via Sequence Embeddings")
    st.markdown("""
    This enhanced application detects potentially fraudulent insurance claims by analyzing 
    multiple aspects of claim sequences, including event patterns, frequencies, and ordering.
    """)
    
    # Load models and sample data
    vectorizer, scaler, classifier, feature_names = load_enhanced_models()
    sample_df = load_sample_data()
    
    if vectorizer is None:
        st.warning("Please train the enhanced models first by running `python models/train_model_enhanced_nosmote.py`")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Pattern Explanation", "How It Works"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Input Claim Sequence")
            
            # Initialize session state for events
            if 'events' not in st.session_state:
                st.session_state.events = []
            
            # Add event interface
            event_type = st.selectbox("Select event type:", EVENT_TYPES)
            add_event = st.button("Add Event")
            
            if add_event:
                st.session_state.events.append(event_type)
                st.success(f"Added: {event_type}")
            
            # Display current sequence
            if st.session_state.events:
                st.subheader("Current Sequence:")
                for i, event in enumerate(st.session_state.events, 1):
                    st.write(f"{i}. {event}")
                
                # Clear sequence button
                if st.button("Clear Sequence"):
                    st.session_state.events = []
                    st.rerun()
            
            # Quick load buttons
            st.subheader("Quick Load Examples")
            
            col1a, col2a = st.columns(2)
            with col1a:
                if st.button("Load Fraud Example"):
                    st.session_state.events = ["First Notice of Loss", "Payment Processing", "Claim Closed"]
                    st.rerun()
            
            with col2a:
                if st.button("Load Legitimate Example"):
                    st.session_state.events = [
                        "First Notice of Loss", "Initial Assessment", "Damage Documentation", 
                        "Repair Authorization", "Repair In Progress", "Final Assessment", 
                        "Payment Processing", "Claim Closed"
                    ]
                    st.rerun()
        
        with col2:
            st.header("Analysis Results")
            
            if st.session_state.events:
                # Create features from the sequence
                features = create_advanced_features_from_sequence(
                    st.session_state.events, vectorizer, feature_names
                )
                
                # Scale features
                scaled_features = scaler.transform(features)
                
                # Make prediction
                prediction = classifier.predict(scaled_features)[0]
                probability = classifier.predict_proba(scaled_features)[0][1]
                
                # Analyze pattern for specific indicators
                pattern_indicators = analyze_sequence_pattern(st.session_state.events)
                
                # Display results
                st.subheader("Prediction:")
                if prediction == 1:
                    st.error(f"🚨 Fraudulent Claim (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ Legitimate Claim (Probability: {1-probability:.2%})")
                
                # Show probability gauge
                st.subheader("Fraud Probability:")
                st.progress(float(probability))
                st.write(f"{probability:.2%}")
                
                # Show pattern analysis
                if pattern_indicators:
                    st.subheader("Pattern Analysis:")
                    for indicator in pattern_indicators:
                        st.warning(f"⚠️ {indicator}")
                else:
                    st.info("No obvious fraud indicators detected in the pattern.")
                
                # Show confidence level
                confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
                st.write(f"**Confidence:** {confidence}")
            
            else:
                st.info("Please add events to the sequence to analyze.")
    
    with tab2:
        st.header("Fraud Pattern Explanation")
        
        st.subheader("Common Fraud Patterns:")
        
        patterns = [
            {
                "pattern": "Short Sequence",
                "example": ["First Notice of Loss", "Payment Processing", "Claim Closed"],
                "explanation": "Skipping essential steps like assessment and documentation",
                "detection": "Sequence length < 4 events"
            },
            {
                "pattern": "Suspicious Repeats",
                "example": ["First Notice of Loss", "Expert Consultation", "Expert Consultation", "Payment Processing"],
                "explanation": "Excessive repetitions of investigation events without resolution",
                "detection": "Same event repeated > 2 times"
            },
            {
                "pattern": "Missing Critical Events",
                "example": ["First Notice of Loss", "Initial Assessment", "Payment Processing"],
                "explanation": "Skipping essential documentation or validation steps",
                "detection": "Missing Damage Documentation or Final Assessment"
            },
            {
                "pattern": "Illogical Ordering",
                "example": ["First Notice of Loss", "Payment Processing", "Damage Documentation"],
                "explanation": "Payment processed before assessment or documentation",
                "detection": "Payment before Assessment/Documentation"
            },
            {
                "pattern": "Investigation-Only",
                "example": ["First Notice of Loss", "Police Report Filed", "Witness Statement", "Payment Processing"],
                "explanation": "Only investigation events without repair/assessment steps",
                "detection": "No repair/assessment events after investigation"
            }
        ]
        
        for pattern in patterns:
            with st.expander(f"{pattern['pattern']} Pattern"):
                st.write("**Example:**")
                for i, event in enumerate(pattern['example'], 1):
                    st.write(f"{i}. {event}")
                
                st.write(f"**Explanation:** {pattern['explanation']}")
                st.write(f"**Detection:** {pattern['detection']}")
    
    with tab3:
        st.header("How the Enhanced System Works:")
        st.markdown("""
        ## Enhanced Detection Approach
        
        This system uses multiple techniques to improve fraud detection accuracy:
        
        1. **Advanced Feature Engineering**:
           - Event frequency counts (expert consultations, witness statements, etc.)
           - Sequence length analysis
           - Event ordering patterns
           - Presence of critical events
        
        2. **TF-IDF with N-grams**:
           - Captures patterns of 1-3 consecutive events
           - Identifies common fraudulent sequences
        
        3. **Hyperparameter Tuning**:
           - Optimizes model parameters for best performance
           - Uses cross-validation to ensure robustness
        
        4. **Pattern Analysis**:
           - Rule-based checks for known fraud patterns
           - Combines ML predictions with heuristic rules
        
        5. **Class Weighting**:
           - Gives more importance to the minority class (fraudulent claims)
           - Helps handle the natural imbalance in the data
        
        ## Why No System is 100% Accurate
        
        1. **Adversarial Nature**: Fraudsters constantly adapt their strategies
        2. **Edge Cases**: Some legitimate claims may look suspicious due to unusual circumstances
        3. **Data Limitations**: Synthetic data can't capture all real-world complexity
        4. **Model Limitations**: All ML models have inherent limitations and trade-offs
        
        ## Improving Real-World Accuracy
        
        To improve accuracy in production:
        
        1. **Use Real Data**: Train on actual historical claim data
        2. **Add More Features**: Incorporate financial, temporal, and claimant data
        3. **Ensemble Methods**: Combine multiple models for better performance
        4. **Human Review**: Use the system as a screening tool, not a final decision-maker
        5. **Continuous Learning**: Regularly update the model with new data
        """)

if __name__ == "__main__":
    main()