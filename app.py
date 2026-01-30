import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Page Configuration
st.set_page_config(
    page_title="AI Population & Resource Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LinkedIn-like CSS Styling
def linkedin_style():
    st.markdown("""
    <style>
    /* Main LinkedIn-like Theme */
    .main {
        background-color: #f3f2ef;
    }
    
    .stApp {
        background-color: #f3f2ef;
    }
    
    /* Header Styling */
    .header-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #0077b5;
    }
    
    /* Card Styling */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #0077b5;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 24px;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #005582;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,119,181,0.3);
    }
    
    /* Secondary Button */
    .secondary-button {
        background-color: #f3f2ef !important;
        color: #000000 !important;
        border: 1px solid #0077b5 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #000000;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0077b5 !important;
        color: #ffffff !important;
        border-bottom: 3px solid #0077b5;
    }
    
    /* Data Editor Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #0077b5;
    }
    
    /* Custom Divider */
    .divider {
        border-top: 2px solid #e0e0e0;
        margin: 30px 0;
    }
    
    /* Success Message */
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    
    /* Warning Message */
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    
    /* Error Message */
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    
    /* Info Box */
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'errors' not in st.session_state:
        st.session_state.errors = []
    if 'warnings' not in st.session_state:
        st.session_state.warnings = []

# Generate template CSV
def generate_template():
    template_data = {
        'Year': [2023, 2023, 2023],
        'GeoID': ['NG-LAG', 'NG-ABJ', 'NG-KAN'],
        'Region': ['Lagos', 'Abuja', 'Kano'],
        'GeoType': ['State', 'Capital', 'State'],
        'Population': [21500000, 3500000, 16000000],
        'Birth_Rate': [28.5, 32.1, 35.6],
        'Death_Rate': [8.2, 7.8, 9.1],
        'Net_Migration': [150000, 50000, -25000],
        'Urbanization_Rate': [88.5, 92.3, 65.4],
        'Refugee_Influx': [1200, 500, 300],
        'Age_0_4_Pct': [12.3, 13.5, 15.2],
        'Age_5_17_Pct': [25.6, 26.8, 28.4],
        'Age_18_45_Pct': [45.8, 44.3, 42.1],
        'Age_46_64_Pct': [12.5, 11.8, 10.3],
        'Age_65Plus_Pct': [3.8, 3.6, 4.0],
        'GDP_Per_Capita': [2800, 5200, 1600],
        'Inflation_Rate': [24.1, 22.8, 25.3],
        'Unemployment_Rate': [33.3, 28.5, 35.6],
        'Youth_Unemployment_Pct': [53.4, 48.9, 56.7],
        'Informal_Economy_Pct': [65.3, 55.8, 72.4],
        'Primary_School_Enrollment': [85.6, 92.3, 78.9],
        'Secondary_School_Enrollment': [68.4, 82.1, 61.5],
        'University_Enrollment': [25.6, 42.3, 18.7],
        'STEM_Graduates_Pct': [32.5, 38.9, 28.4],
        'Hospital_Beds_Per_1000': [0.8, 1.2, 0.6],
        'Physicians_Per_1000': [0.4, 0.9, 0.3],
        'Vaccination_Coverage': [65.8, 78.9, 58.4],
        'Maternal_Mortality_Rate': [512, 398, 625],
        'Electricity_Access_Pct': [86.5, 94.2, 72.3],
        'Clean_Water_Access_Pct': [78.9, 89.4, 65.8],
        'Sanitation_Access_Pct': [65.4, 82.1, 54.6],
        'Internet_Penetration_Rate': [55.6, 78.9, 42.3],
        'Arable_Land_Pct': [32.5, 18.9, 45.6],
        'Crop_Yield_Index': [1.2, 1.0, 1.1],
        'Food_Security_Score': [58.9, 65.4, 52.3],
        'Protein_Intake_Grams': [45.6, 52.3, 38.9],
        'Road_Density_KM2': [0.45, 0.68, 0.32],
        'Rail_Network_KM': [350, 280, 150],
        'Port_Capacity_TEU': [2800000, 0, 0],
        'Air_Traffic_Volume': [15800000, 12500000, 4500000],
        'Manufacturing_Value_Added_Pct': [8.9, 6.5, 7.2],
        'Export_Diversity_Index': [42.3, 38.9, 35.6],
        'Foreign_Direct_Investment': [3500, 2800, 1200],
        'Carbon_Emissions_Per_Capita': [0.6, 1.2, 0.4],
        'Forest_Cover_Pct': [12.5, 28.9, 8.4],
        'Disaster_Risk_Index': [5.6, 3.2, 6.8],
        'Police_Per_100000': [180, 220, 150],
        'Prison_Population_Pct': [0.12, 0.08, 0.15],
        'Terrorism_Threat_Index': [6.8, 4.2, 7.5],
        'Government_Effectiveness_Score': [45.6, 52.3, 38.9],
        'Corruption_Perception_Index': [24, 28, 22],
        'Social_Cohesion_Index': [65.8, 72.3, 58.9]
    }
    return pd.DataFrame(template_data)

# Data Validation Function
def validate_data(df):
    errors = []
    warnings = []
    
    # Required columns
    required_columns = [
        'Year', 'GeoID', 'Region', 'GeoType', 'Population',
        'Birth_Rate', 'Death_Rate', 'Net_Migration'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check data types
    numeric_columns = [
        'Population', 'Birth_Rate', 'Death_Rate', 'Net_Migration',
        'Urbanization_Rate', 'GDP_Per_Capita', 'Inflation_Rate'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    warnings.append(f"Converted column {col} to numeric")
                except:
                    errors.append(f"Column {col} contains non-numeric values")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    for col, count in missing_values.items():
        if count > 0:
            warnings.append(f"Column {col} has {count} missing values")
    
    # Validate ranges
    if 'Birth_Rate' in df.columns:
        if (df['Birth_Rate'] < 0).any() or (df['Birth_Rate'] > 100).any():
            warnings.append("Birth rate should be between 0 and 100")
    
    if 'Death_Rate' in df.columns:
        if (df['Death_Rate'] < 0).any() or (df['Death_Rate'] > 100).any():
            warnings.append("Death rate should be between 0 and 100")
    
    return errors, warnings

# Data Preprocessing Function
def preprocess_data(df):
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with median
    imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
    
    # Normalize numeric features (optional, based on model needs)
    scaler = StandardScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
    
    return df_clean, label_encoders

# Train Random Forest Model
def train_random_forest(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return rf, mae, rmse, r2, y_test, y_pred

# Resource Allocation Recommendation
def recommend_resources(df, predictions):
    recommendations = []
    
    for idx, row in df.iterrows():
        region = row['Region'] if 'Region' in df.columns else f"Region_{idx}"
        
        # Analyze needs based on metrics
        needs = []
        
        if 'Hospital_Beds_Per_1000' in df.columns and row['Hospital_Beds_Per_1000'] < 1:
            needs.append(('Healthcare Facilities', 'High', 
                         f"Current beds: {row['Hospital_Beds_Per_1000']:.2f} per 1000, WHO recommends minimum 3"))
        
        if 'Primary_School_Enrollment' in df.columns and row['Primary_School_Enrollment'] < 90:
            needs.append(('Educational Infrastructure', 'Medium',
                         f"Enrollment rate: {row['Primary_School_Enrollment']:.1f}%"))
        
        if 'Clean_Water_Access_Pct' in df.columns and row['Clean_Water_Access_Pct'] < 80:
            needs.append(('Water Supply Systems', 'High',
                         f"Access: {row['Clean_Water_Access_Pct']:.1f}%"))
        
        if 'Unemployment_Rate' in df.columns and row['Unemployment_Rate'] > 30:
            needs.append(('Job Creation Programs', 'High',
                         f"Unemployment: {row['Unemployment_Rate']:.1f}%"))
        
        if 'Road_Density_KM2' in df.columns and row['Road_Density_KM2'] < 0.5:
            needs.append(('Transport Infrastructure', 'Medium',
                         f"Road density: {row['Road_Density_KM2']:.2f} km/km²"))
        
        # Prioritize based on severity
        needs.sort(key=lambda x: 0 if x[1] == 'High' else 1 if x[1] == 'Medium' else 2)
        
        recommendations.append({
            'Region': region,
            'Predicted_Population_Growth': predictions[idx] if idx < len(predictions) else 'N/A',
            'Priority_Needs': needs[:3],  # Top 3 needs
            'Timeline': 'Short-term' if len([n for n in needs if n[1] == 'High']) > 1 else 'Medium-term'
        })
    
    return recommendations

# Create Visualization Dashboard
def create_visualizations(df, predictions=None):
    visualizations = []
    
    # 1. Population Distribution
    if 'Population' in df.columns and 'Region' in df.columns:
        fig1 = px.bar(df.nlargest(10, 'Population'), 
                     x='Region', y='Population',
                     title='Top 10 Regions by Population',
                     color='Population',
                     color_continuous_scale='viridis')
        visualizations.append(fig1)
    
    # 2. Birth vs Death Rates
    if 'Birth_Rate' in df.columns and 'Death_Rate' in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Region'], y=df['Birth_Rate'], 
                                 mode='markers+lines', name='Birth Rate',
                                 line=dict(color='green', width=2)))
        fig2.add_trace(go.Scatter(x=df['Region'], y=df['Death_Rate'], 
                                 mode='markers+lines', name='Death Rate',
                                 line=dict(color='red', width=2)))
        fig2.update_layout(title='Birth vs Death Rates by Region',
                          xaxis_title='Region',
                          yaxis_title='Rate (%)',
                          template='plotly_white')
        visualizations.append(fig2)
    
    # 3. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 5:
        corr_matrix = df[numeric_cols[:10]].corr()
        fig3 = px.imshow(corr_matrix,
                        title='Feature Correlation Heatmap',
                        color_continuous_scale='RdBu',
                        aspect='auto')
        visualizations.append(fig3)
    
    # 4. Resource Gaps Analysis
    resource_cols = ['Hospital_Beds_Per_1000', 'Primary_School_Enrollment', 
                    'Clean_Water_Access_Pct', 'Electricity_Access_Pct']
    available_cols = [col for col in resource_cols if col in df.columns]
    
    if available_cols:
        fig4 = make_subplots(rows=2, cols=2, 
                           subplot_titles=available_cols[:4])
        
        for i, col in enumerate(available_cols[:4]):
            row = (i // 2) + 1
            col_num = (i % 2) + 1
            
            fig4.add_trace(
                go.Bar(x=df['Region'], y=df[col], name=col),
                row=row, col=col_num
            )
        
        fig4.update_layout(height=600, title_text="Resource Access by Region",
                          showlegend=False)
        visualizations.append(fig4)
    
    return visualizations

# Main Application
def main():
    # Apply LinkedIn-like styling
    linkedin_style()
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 style="color: #0077b5; margin-bottom: 10px;">🌍 AI Population Growth & Resource Planning System</h1>
        <p style="color: #666666; font-size: 16px;">Advanced Analytics for Demographic Forecasting and Infrastructure Planning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Tabs
    tabs = st.tabs(["📥 Data Management", "🔍 Data Analysis", "🤖 AI Modeling", "📊 Visualizations", "🎯 Recommendations"])
    
    # Tab 1: Data Management
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Data Upload & Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Download Template")
            st.write("Download our standardized template to ensure data compatibility")
            
            if st.button("📥 Download CSV Template", key="download_template"):
                template_df = generate_template()
                csv = template_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="population_template.csv">Click here to download</a>'
                st.markdown(f'<div class="success-box">Template ready! {href}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📤 Upload Your Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_uploaded = True
                    
                    st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), use_container_width=True, height=300)
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Validation Section
        if st.session_state.data_uploaded:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("🔍 Data Validation & Quality Check")
            
            errors, warnings = validate_data(st.session_state.df)
            st.session_state.errors = errors
            st.session_state.warnings = warnings
            
            col1, col2 = st.columns(2)
            
            with col1:
                if errors:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.subheader("❌ Critical Errors")
                    for error in errors:
                        st.write(f"• {error}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.subheader("✅ No Critical Errors Found")
                    st.write("Your data passes basic validation checks")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if warnings:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.subheader("⚠️ Warnings & Suggestions")
                    for warning in warnings:
                        st.write(f"• {warning}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Data Editor for manual corrections
            st.subheader("✏️ Edit Data (if needed)")
            edited_df = st.data_editor(st.session_state.df, 
                                      use_container_width=True,
                                      height=400,
                                      num_rows="dynamic")
            
            if st.button("🔄 Update Data", key="update_data"):
                st.session_state.df = edited_df
                st.success("Data updated successfully!")
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Data Analysis
    with tabs[1]:
        if st.session_state.data_uploaded:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("📈 Statistical Analysis & Preprocessing")
            
            # Basic Statistics
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
            
            # Missing Values Analysis
            st.subheader("🔍 Missing Values Analysis")
            missing_data = st.session_state.df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing_Values': missing_data.values,
                'Percentage': (missing_data.values / len(st.session_state.df)) * 100
            })
            st.dataframe(missing_df[missing_df['Missing_Values'] > 0], 
                        use_container_width=True)
            
            # Data Preprocessing Options
            st.subheader("⚙️ Data Preprocessing Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Auto-Preprocess Data", key="auto_preprocess"):
                    with st.spinner("Processing data..."):
                        processed_df, encoders = preprocess_data(st.session_state.df)
                        st.session_state.df = processed_df
                        st.session_state.processed = True
                        st.success("Data preprocessing completed!")
                        st.rerun()
            
            with col2:
                if st.button("🧹 Handle Missing Values", key="handle_missing"):
                    # Simple imputation
                    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                    st.session_state.df[numeric_cols] = st.session_state.df[numeric_cols].fillna(
                        st.session_state.df[numeric_cols].median())
                    st.success("Missing values handled!")
            
            with col3:
                if st.button("📏 Normalize Data", key="normalize"):
                    # Standardize numeric columns
                    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                    scaler = StandardScaler()
                    st.session_state.df[numeric_cols] = scaler.fit_transform(
                        st.session_state.df[numeric_cols])
                    st.success("Data normalized!")
            
            # Correlation Analysis
            if st.session_state.processed:
                st.subheader("🔗 Feature Correlation Analysis")
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = st.session_state.df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                               cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: AI Modeling
    with tabs[2]:
        if st.session_state.data_uploaded:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("🤖 Machine Learning Model Training")
            
            # Feature Selection
            st.subheader("🎯 Select Features for Prediction")
            
            all_columns = st.session_state.df.columns.tolist()
            numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Target variable selection
            target_options = ['Population', 'Birth_Rate', 'Death_Rate', 'Net_Migration']
            available_targets = [col for col in target_options if col in numeric_columns]
            
            if available_targets:
                target_variable = st.selectbox(
                    "Select target variable for prediction:",
                    available_targets,
                    key="target_select"
                )
                
                # Feature selection
                feature_options = [col for col in numeric_columns if col != target_variable]
                selected_features = st.multiselect(
                    "Select features for the model:",
                    feature_options,
                    default=feature_options[:min(10, len(feature_options))]
                )
                
                if selected_features and target_variable:
                    # Prepare data
                    X = st.session_state.df[selected_features]
                    y = st.session_state.df[target_variable]
                    
                    # Train model
                    if st.button("🚀 Train Random Forest Model", key="train_model"):
                        with st.spinner("Training model... This may take a moment."):
                            progress_bar = st.progress(0)
                            
                            for i in range(100):
                                progress_bar.progress(i + 1)
                            
                            model, mae, rmse, r2, y_test, y_pred = train_random_forest(X, y)
                            st.session_state.model = model
                            st.session_state.model_trained = True
                            st.session_state.metrics = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
                            st.session_state.y_test = y_test
                            st.session_state.y_pred = y_pred
                            
                            st.success("Model training completed!")
                
                # Display Model Performance
                if st.session_state.model_trained:
                    st.subheader("📊 Model Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Mean Absolute Error", 
                                 f"{st.session_state.metrics['MAE']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Root Mean Squared Error", 
                                 f"{st.session_state.metrics['RMSE']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("R² Score", 
                                 f"{st.session_state.metrics['R²']:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feature Importance
                    st.subheader("📈 Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': st.session_state.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance.head(10),
                                x='Importance', y='Feature',
                                orientation='h',
                                title='Top 10 Most Important Features',
                                color='Importance',
                                color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actual vs Predicted
                    st.subheader("📉 Actual vs Predicted Values")
                    comparison_df = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': st.session_state.y_pred
                    }).head(20)
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=comparison_df.index, 
                                             y=comparison_df['Actual'],
                                             mode='lines+markers',
                                             name='Actual',
                                             line=dict(color='blue', width=2)))
                    fig2.add_trace(go.Scatter(x=comparison_df.index,
                                             y=comparison_df['Predicted'],
                                             mode='lines+markers',
                                             name='Predicted',
                                             line=dict(color='red', width=2)))
                    fig2.update_layout(title='Actual vs Predicted Values Comparison',
                                      xaxis_title='Sample Index',
                                      yaxis_title=target_variable,
                                      template='plotly_white')
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Visualizations
    with tabs[3]:
        if st.session_state.data_uploaded:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("📊 Interactive Visualizations")
            
            # Generate visualizations
            visualizations = create_visualizations(st.session_state.df)
            
            for i, fig in enumerate(visualizations):
                st.plotly_chart(fig, use_container_width=True)
                if i < len(visualizations) - 1:
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Custom Visualization Options
            st.subheader("🛠️ Create Custom Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-axis variable:", 
                                     st.session_state.df.columns.tolist())
            
            with col2:
                y_axis = st.selectbox("Select Y-axis variable:", 
                                     st.session_state.df.select_dtypes(include=[np.number]).columns.tolist())
            
            if x_axis and y_axis:
                chart_type = st.selectbox("Select chart type:", 
                                         ["Scatter", "Line", "Bar", "Histogram"])
                
                if st.button("Generate Custom Chart"):
                    if chart_type == "Scatter":
                        fig = px.scatter(st.session_state.df, x=x_axis, y=y_axis,
                                        color=y_axis, size=y_axis,
                                        hover_data=st.session_state.df.columns.tolist()[:5])
                    elif chart_type == "Line":
                        fig = px.line(st.session_state.df, x=x_axis, y=y_axis)
                    elif chart_type == "Bar":
                        fig = px.bar(st.session_state.df, x=x_axis, y=y_axis)
                    else:
                        fig = px.histogram(st.session_state.df, x=y_axis)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Recommendations
    with tabs[4]:
        if st.session_state.data_uploaded:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("🎯 Resource Allocation Recommendations")
            
            if st.session_state.model_trained:
                # Generate predictions for resource allocation
                if 'Population' in st.session_state.df.columns:
                    # Use the model to predict population growth
                    X = st.session_state.df[selected_features]
                    predictions = st.session_state.model.predict(X)
                    
                    # Generate recommendations
                    recommendations = recommend_resources(st.session_state.df, predictions)
                    
                    # Display recommendations
                    for rec in recommendations:
                        st.markdown(f"### 🎯 {rec['Region']}")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Priority Needs:**")
                            for need in rec['Priority_Needs']:
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                    <strong>{need[0]}</strong> - Priority: <span style="color: {'#dc3545' if need[1]=='High' else '#ffc107' if need[1]=='Medium' else '#28a745'}">{need[1]}</span><br>
                                    <small>{need[2]}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Timeline", rec['Timeline'])
                            if 'Predicted_Population_Growth' in rec and rec['Predicted_Population_Growth'] != 'N/A':
                                st.metric("Predicted Growth", f"{rec['Predicted_Population_Growth']:.2f}%")
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # Summary Statistics
                    st.subheader("📋 Summary Analysis")
                    
                    summary_stats = {
                        'Total Regions Analyzed': len(recommendations),
                        'Regions with High Priority Needs': len([r for r in recommendations if any(n[1]=='High' for n in r['Priority_Needs'])]),
                        'Average Predicted Growth': np.mean([float(r['Predicted_Population_Growth']) for r in recommendations if r['Predicted_Population_Growth'] != 'N/A']),
                        'Most Common Need': max(set([n[0] for r in recommendations for n in r['Priority_Needs']]), 
                                               key=[n[0] for r in recommendations for n in r['Priority_Needs']].count)
                    }
                    
                    cols = st.columns(4)
                    for idx, (key, value) in enumerate(summary_stats.items()):
                        with cols[idx % 4]:
                            st.metric(key, str(value))
                    
                    # Export Recommendations
                    st.subheader("📤 Export Recommendations")
                    
                    if st.button("💾 Export as CSV"):
                        rec_df = pd.DataFrame(recommendations)
                        csv = rec_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="resource_recommendations.csv">Download Recommendations CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.info("Population data is required for resource allocation recommendations.")
            else:
                st.warning("Please train the model first in the AI Modeling tab to generate recommendations.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3 style="margin: 0;">System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Status Indicators
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.metric("Data Loaded", 
                     "✅" if st.session_state.data_uploaded else "❌",
                     delta=None)
        
        with status_col2:
            st.metric("Model Trained",
                     "✅" if st.session_state.model_trained else "❌",
                     delta=None)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Quick Stats
        if st.session_state.data_uploaded:
            st.markdown("### 📊 Quick Stats")
            st.write(f"**Rows:** {len(st.session_state.df):,}")
            st.write(f"**Columns:** {len(st.session_state.df.columns)}")
            st.write(f"**Missing Values:** {st.session_state.df.isnull().sum().sum():,}")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Information Panel
        st.markdown("### ℹ️ How to Use")
        st.info("""
        1. **Download** the template CSV
        2. **Upload** your demographic data
        3. **Validate** and preprocess data
        4. **Train** the AI model
        5. **Analyze** visualizations
        6. **Implement** recommendations
        """)
        
        # Data Sources
        st.markdown("### 📚 Data Sources")
        st.write("""
        • National Bureau of Statistics (NBS)
        • National Population Commission (NPC)
        • World Bank
        • UN Population Division
        • Local Survey Data
        """)
        
        # Contact/Help
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🆘 Need Help?")
        st.markdown("""
        <div class="info-box">
        Contact support for:
        • Data formatting issues
        • Model interpretation
        • Implementation guidance
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.caption("AI Population Growth & Resource Planning System v1.0")
        st.caption("© 2024 All rights reserved")

if __name__ == "__main__":
    main()