# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CyberFedDefender - Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Header
st.markdown('<h1 class="main-header">🛡️ CyberFedDefender</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Network Intrusion Detection System using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=100)
    st.title("Navigation")
    
    menu_options = ["📤 Data Upload", "📊 EDA", "🤖 Model Training", "🎯 Prediction", "📈 Model Comparison"]
    choice = st.radio("Go to", menu_options)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This application uses machine learning to detect network intrusions. Upload your cybersecurity dataset and train multiple models to identify malicious activities.")
    
    if st.button("🔄 Reset App"):
        st.session_state.models_trained = False
        st.session_state.df = None
        st.session_state.results_df = None
        st.session_state.models = None
        st.session_state.scalers = {}
        st.session_state.label_encoders = {}
        st.rerun()

# Data Upload Section
if choice == "📤 Data Upload":
    st.header("📤 Upload Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("✅ Dataset loaded successfully!")
                
                # Display dataset info
                st.subheader("Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                st.subheader("First 5 Rows")
                st.dataframe(df.head())
                
                st.subheader("Dataset Info")
                buffer = []
                df.info(buf=buffer)
                st.text("\n".join(buffer))
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.subheader("Sample Format")
        sample_data = {
            'Timestamp': ['2023-01-01 00:00:01'],
            'Source_IP': ['192.168.1.1'],
            'Destination_IP': ['10.0.0.1'],
            'Protocol': ['TCP'],
            'Packet_Length': [1500],
            'Duration': [0.5],
            'Bytes_Sent': [1024],
            'Bytes_Received': [2048],
            'Flow_Packets/s': [100],
            'Flow_Bytes/s': [50000],
            'Avg_Packet_Size': [1024],
            'Label': [0]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        st.markdown("### Expected Columns")
        st.info("""
        - Timestamp
        - Source_IP
        - Destination_IP
        - Protocol
        - Packet_Length
        - Duration
        - Bytes_Sent
        - Bytes_Received
        - Flow_Packets/s
        - Flow_Bytes/s
        - Avg_Packet_Size
        - Label (0=Normal, 1=Attack)
        """)

# EDA Section
elif choice == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Data preprocessing for EDA
        cols_to_drop = ['Timestamp', 'Source_IP', 'Destination_IP']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
        
        # Encode Protocol for visualization
        if 'Protocol' in df.columns and df['Protocol'].dtype == 'object':
            le_protocol = LabelEncoder()
            df['Protocol_encoded'] = le_protocol.fit_transform(df['Protocol'])
            st.session_state.label_encoders['protocol'] = le_protocol
        
        # Create tabs for different EDA sections
        tab1, tab2, tab3, tab4 = st.tabs(["Target Distribution", "Feature Distributions", "Box Plots", "Correlation Matrix"])
        
        with tab1:
            st.subheader("Target Variable Distribution")
            if 'Label' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(df, names='Label', title='Label Distribution', 
                                color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    label_counts = df['Label'].value_counts()
                    fig = px.bar(x=label_counts.index, y=label_counts.values, 
                               labels={'x': 'Label', 'y': 'Count'},
                               title='Label Counts', text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                st.markdown("### Statistics")
                normal_pct = (df['Label'] == 0).sum() / len(df) * 100
                attack_pct = (df['Label'] == 1).sum() / len(df) * 100
                col1, col2 = st.columns(2)
                col1.metric("Normal Traffic", f"{normal_pct:.2f}%")
                col2.metric("Attack Traffic", f"{attack_pct:.2f}%")
        
        with tab2:
            st.subheader("Feature Distributions")
            
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in num_cols:
                num_cols.remove('Label')
            
            selected_feature = st.selectbox("Select Feature", num_cols)
            
            if selected_feature:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(df, x=selected_feature, nbins=50, 
                                      title=f'Distribution of {selected_feature}',
                                      marginal='box')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(df, y=selected_feature, title=f'Box Plot of {selected_feature}')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Distribution by Label")
            
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in num_cols:
                num_cols.remove('Label')
            if 'Protocol_encoded' in num_cols:
                num_cols.remove('Protocol_encoded')
            
            selected_feature = st.selectbox("Select Feature for Box Plot", num_cols, key='box_select')
            
            if selected_feature:
                fig = px.box(df, x='Label', y=selected_feature, 
                           title=f'{selected_feature} Distribution by Label',
                           color='Label', color_discrete_sequence=['green', 'red'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Correlation Matrix")
            
            # Select numerical columns for correlation
            num_df = df.select_dtypes(include=[np.number])
            
            if not num_df.empty:
                corr_matrix = num_df.corr()
                
                fig = px.imshow(corr_matrix, 
                              text_auto='.2f',
                              aspect="auto",
                              color_continuous_scale='RdBu_r',
                              title='Feature Correlation Matrix')
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' section.")

# Model Training Section
elif choice == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Data preprocessing
        with st.expander("Data Preprocessing Options", expanded=True):
            st.markdown("### Drop Non-Predictive Columns")
            cols_to_drop = ['Timestamp', 'Source_IP', 'Destination_IP']
            available_cols = [col for col in cols_to_drop if col in df.columns]
            
            selected_drop_cols = st.multiselect("Select columns to drop", 
                                               available_cols, 
                                               default=available_cols)
            
            if selected_drop_cols:
                df.drop(columns=selected_drop_cols, inplace=True)
                st.success(f"Dropped columns: {selected_drop_cols}")
            
            st.markdown("### Encode Categorical Features")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_cols:
                if col != 'Label':  # Don't encode target if it's categorical
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    st.session_state.label_encoders[col] = le
                    st.info(f"Encoded {col}")
        
        # Train/Test Split Configuration
        st.markdown("### Train/Test Split Configuration")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 30, 20, 1) / 100
        with col2:
            random_state = st.number_input("Random State", value=42, min_value=0, max_value=250)
        
        # Prepare features and target
        if 'Label' in df.columns:
            X = df.drop('Label', axis=1)
            y = df['Label']
            
            # Ensure all features are numeric
            X = X.select_dtypes(include=[np.number])
            
            st.success(f"Features used for training: {len(X.columns)}")
            st.write(X.columns.tolist())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler_std = StandardScaler()
            X_train_std = scaler_std.fit_transform(X_train)
            X_test_std = scaler_std.transform(X_test)
            
            scaler_mm = MinMaxScaler()
            X_train_mm = scaler_mm.fit_transform(X_train)
            X_test_mm = scaler_mm.transform(X_test)
            
            st.session_state.scalers['standard'] = scaler_std
            st.session_state.scalers['minmax'] = scaler_mm
            
            # Model selection
            st.markdown("### Select Models to Train")
            col1, col2 = st.columns(2)
            
            with col1:
                train_gnb = st.checkbox("Gaussian Naive Bayes", value=True)
                train_mnb = st.checkbox("Multinomial Naive Bayes", value=True)
                train_bnb = st.checkbox("Bernoulli Naive Bayes", value=True)
            
            with col2:
                train_lr = st.checkbox("Logistic Regression", value=True)
                train_rf = st.checkbox("Random Forest", value=True)
            
            # Training button
            if st.button("🚀 Train Selected Models", type="primary"):
                with st.spinner("Training models... This may take a moment."):
                    models = {}
                    results = {}
                    
                    if train_gnb:
                        gnb = GaussianNB()
                        gnb.fit(X_train_std, y_train)
                        y_pred = gnb.predict(X_test_std)
                        results['Gaussian NB'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred)
                        }
                        models['Gaussian NB'] = gnb
                    
                    if train_mnb:
                        mnb = MultinomialNB()
                        mnb.fit(X_train_mm, y_train)
                        y_pred = mnb.predict(X_test_mm)
                        results['Multinomial NB'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred)
                        }
                        models['Multinomial NB'] = mnb
                    
                    if train_bnb:
                        bnb = BernoulliNB()
                        bnb.fit(X_train_std, y_train)
                        y_pred = bnb.predict(X_test_std)
                        results['Bernoulli NB'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred)
                        }
                        models['Bernoulli NB'] = bnb
                    
                    if train_lr:
                        lr = LogisticRegression(max_iter=1000, random_state=random_state)
                        lr.fit(X_train_std, y_train)
                        y_pred = lr.predict(X_test_std)
                        results['Logistic Regression'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred)
                        }
                        models['Logistic Regression'] = lr
                    
                    if train_rf:
                        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                        rf.fit(X_train_std, y_train)
                        y_pred = rf.predict(X_test_std)
                        results['Random Forest'] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred)
                        }
                        models['Random Forest'] = rf
                    
                    # Store in session state
                    st.session_state.models_trained = True
                    st.session_state.models = models
                    st.session_state.results_df = pd.DataFrame(results).T
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Display results
                    st.markdown("### Training Results")
                    st.dataframe(st.session_state.results_df.style.highlight_max(axis=0))
        else:
            st.error("Dataset must contain a 'Label' column for training!")
    else:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' section.")

# Prediction Section
elif choice == "🎯 Prediction":
    st.header("🎯 Make Predictions")
    
    if st.session_state.models_trained and st.session_state.models is not None:
        st.markdown("### Select Model for Prediction")
        
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Choose Model", model_names)
        
        if selected_model:
            model = st.session_state.models[selected_model]
            
            # Input methods
            input_method = st.radio("Input Method", ["Manual Input", "Upload Test File"])
            
            if input_method == "Manual Input":
                st.markdown("### Enter Feature Values")
                
                # Get feature names from training data
                feature_names = st.session_state.X_train.columns.tolist()
                
                # Create input fields
                col1, col2 = st.columns(2)
                input_values = {}
                
                for i, feature in enumerate(feature_names):
                    with col1 if i % 2 == 0 else col2:
                        input_values[feature] = st.number_input(
                            f"{feature}", 
                            value=0.0,
                            format="%.2f",
                            key=f"input_{feature}"
                        )
                
                if st.button("🔮 Predict", type="primary"):
                    # Create dataframe from input
                    input_df = pd.DataFrame([input_values])
                    
                    # Scale input
                    if selected_model == 'Multinomial NB':
                        input_scaled = st.session_state.scalers['minmax'].transform(input_df)
                    else:
                        input_scaled = st.session_state.scalers['standard'].transform(input_df)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None
                    
                    # Display result
                    st.markdown("---")
                    st.markdown("### Prediction Result")
                    
                    if prediction == 0:
                        st.success("✅ **Normal Traffic**")
                    else:
                        st.error("⚠️ **Attack Detected!**")
                    
                    if probability is not None:
                        st.markdown("#### Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': ['Normal (0)', 'Attack (1)'],
                            'Probability': probability
                        })
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                   text_auto='.2%', color='Class',
                                   color_discrete_map={'Normal (0)': 'green', 'Attack (1)': 'red'})
                        st.plotly_chart(fig, use_container_width=True)
            
            else:  # File upload
                uploaded_pred_file = st.file_uploader("Upload CSV for prediction", type="csv")
                
                if uploaded_pred_file is not None:
                    pred_df = pd.read_csv(uploaded_pred_file)
                    st.write("Uploaded Data Preview:", pred_df.head())
                    
                    if st.button("🔮 Predict from File"):
                        # Ensure same features as training
                        feature_names = st.session_state.X_train.columns.tolist()
                        
                        # Check if all required features are present
                        missing_features = set(feature_names) - set(pred_df.columns)
                        if missing_features:
                            st.error(f"Missing features: {missing_features}")
                        else:
                            # Select only required features in correct order
                            pred_features = pred_df[feature_names]
                            
                            # Scale
                            if selected_model == 'Multinomial NB':
                                pred_scaled = st.session_state.scalers['minmax'].transform(pred_features)
                            else:
                                pred_scaled = st.session_state.scalers['standard'].transform(pred_features)
                            
                            # Predict
                            predictions = model.predict(pred_scaled)
                            
                            # Add predictions to dataframe
                            pred_df['Prediction'] = predictions
                            pred_df['Prediction_Label'] = pred_df['Prediction'].map({0: 'Normal', 1: 'Attack'})
                            
                            # Display results
                            st.markdown("### Prediction Results")
                            st.dataframe(pred_df)
                            
                            # Download button
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
    else:
        st.warning("⚠️ Please train models first in the 'Model Training' section.")

# Model Comparison Section
elif choice == "📈 Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        
        # Metrics comparison
        st.markdown("### Performance Metrics")
        
        # Create bar chart
        fig = go.Figure()
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=results_df.index,
                y=results_df[metric],
                text=results_df[metric].round(3),
                textposition='auto',
                marker_color=color
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### Detailed Metrics Table")
        styled_df = results_df.style.highlight_max(axis=0).format("{:.3f}")
        st.dataframe(styled_df)
        
        # Best model recommendation
        st.markdown("### 🏆 Best Model Recommendation")
        
        # Calculate average score across all metrics
        avg_scores = results_df.mean(axis=1)
        best_model = avg_scores.idxmax()
        best_score = avg_scores.max()
        
        st.success(f"**{best_model}** achieves the highest average performance ({best_score:.3f}) across all metrics.")
        
        # Radar chart for model comparison
        st.markdown("### Radar Chart Comparison")
        
        fig = go.Figure()
        
        for model in results_df.index:
            fig.add_trace(go.Scatterpolar(
                r=results_df.loc[model, metrics].values,
                theta=[m.capitalize() for m in metrics],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:

        st.warning("⚠️ No model results available. Please train models first.")