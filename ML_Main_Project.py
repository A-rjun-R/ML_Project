import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Burnout Risk Prediction",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Employee Burnout Risk Prediction System")
st.markdown("---")

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------
@st.cache_data
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def replace_question_and_cast(df, cols):
    """
    Replace common placeholders with NaN and try to cast numeric-like columns.
    """
    placeholders = ['?', 'NA', 'N/A', '-', '', 'null', None, 'nan', 'NaN', '--', 'n/a']
    for c in cols:
        if c in df.columns:
            # replace placeholders with actual np.nan
            df[c] = df[c].replace(placeholders, np.nan)
            # strip whitespace for object types
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})
            # attempt to coerce to numeric where possible (keep strings if truly non-numeric)
            coerced = pd.to_numeric(df[c], errors='coerce')
            # if coercion yields at least one non-null numeric, use numeric series; else leave original
            if coerced.notna().any():
                df[c] = coerced
    return df

def set_zeros_to_nan(df, cols):
    for c in cols:
        if c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                # coerce to numeric for the comparison, but don't force replacement of strings now
                coerced = pd.to_numeric(df[c], errors='coerce')
                mask_zero = coerced == 0
                if mask_zero.any():
                    # Replace the matching rows in the original column with NaN
                    df.loc[mask_zero, c] = np.nan
    return df

def impute_columns(df):
    """
    Impute columns using rules. Numeric statistics are computed on a numeric-coerced view
    to avoid errors if a column contains stray strings.
    """
    impute_rules = {
        'week': ('mode', None),
        'hours_worked': ('mean', None),
        'work_life_balance_score': ('mean', None),
        'meetings_count': ('mode', None),
        'remote_days': ('mode', None),
        'sleep_hours': ('mean', None),
        'productivity_score': ('mean', None),
        'overtime_hours': ('mean', None),
        'stress_level': ('mean', None),
        'peer_feedback': ('mean', None),
        'burnout_risk': ('mode', None)
    }

    for col, (method, value) in impute_rules.items():
        if col not in df.columns:
            continue

        series = df[col]

        # create numeric view for computing numeric stats (coerce errors to NaN)
        numeric_view = pd.to_numeric(series, errors='coerce')

        if method == 'mode':
            # try numeric mode first, then object mode, then fallback 0
            mode_val = None
            if numeric_view.notna().any():
                mode_vals = numeric_view.mode()
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
            if mode_val is None:
                obj_mode = series.mode(dropna=True)
                mode_val = obj_mode.iloc[0] if not obj_mode.empty else 0
            df[col].fillna(mode_val, inplace=True)

        elif method == 'mean':
            mean_val = numeric_view.mean()
            if pd.isna(mean_val):
                # fallback to object->mode or 0
                obj_mode = series.mode(dropna=True)
                mean_val = obj_mode.iloc[0] if not obj_mode.empty else 0
            # ensure column becomes numeric after imputation
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(mean_val)

        elif method == 'median':
            median_val = numeric_view.median()
            if pd.isna(median_val):
                obj_mode = series.mode(dropna=True)
                median_val = obj_mode.iloc[0] if not obj_mode.empty else 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(median_val)

        elif value is not None:
            df[col].fillna(value, inplace=True)

    return df

def remove_outliers_iterative(df, columns_to_check):
    outlier_summary = {}
    cleaned = df.copy()
    for col in columns_to_check:
        if col not in cleaned.columns:
            continue
        if not pd.api.types.is_numeric_dtype(cleaned[col]):
            # try to coerce temporarily for outlier detection
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

        if cleaned[col].dropna().empty:
            outlier_summary[col] = {
                "lower_bound": None,
                "upper_bound": None,
                "outlier_count": 0
            }
            continue

        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (cleaned[col] >= lower) & (cleaned[col] <= upper)
        outlier_count = int((~mask).sum())
        outlier_summary[col] = {
            "lower_bound": float(lower) if pd.notna(lower) else None,
            "upper_bound": float(upper) if pd.notna(upper) else None,
            "outlier_count": outlier_count
        }
        cleaned = cleaned[mask].copy()
        # if cleaned becomes empty, stop further filtering
        if cleaned.empty:
            break
    return cleaned, outlier_summary

def train_and_evaluate(X, y, test_size=0.3, random_state=1, n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "feature_names": X.columns.tolist()
    }

# -----------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'model_result' not in st.session_state:
    st.session_state.model_result = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Data Overview", "🔧 Data Preprocessing",
     "📈 EDA", "🤖 Model Training", "🎯 Predict Burnout Risk", "💾 Download Model"]
)

# File upload in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])
use_example = st.sidebar.checkbox("Use example dataset", value=False)

if uploaded_file is not None:
    st.session_state.df_original = load_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
elif use_example:
    # Create example dataset
    example_df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "week": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "hours_worked": [40, 42, 38, 45, 45, 43, 39, 41, 44, 44],
        "work_life_balance_score": [7, 6, 5, 8, 6, 6, 5, 7, 6, 6],
        "meetings_count": [3, 2, 4, 5, 2, 2, 3, 4, 2, 2],
        "remote_days": [2, 3, 2, 3, 2, 2, 2, 3, 3, 0],
        "overtime_hours": [0, 2, 1, 0, 0, 1, 0, 2, 1, 1],
        "stress_level": [3, 4, 2, 3, 3, 3, 2, 3, 3, 3],
        "sleep_hours": [7, 6, 6, 5, 8, 7, 6, 7, 4, 7],
        "productivity_score": [80, 75, 60, 70, 85, 90, 75, 50, 40, 95],
        "peer_feedback": [4, 5, 3, 3, 4, 4, 5, 4, 3, 4],
        "burnout_risk": [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]
    })
    st.session_state.df_original = example_df
    st.sidebar.info("Using example dataset")

# -----------------------------------
# HOME PAGE
# -----------------------------------
if page == "🏠 Home":
    st.header("Welcome to Burnout Risk Prediction System")
    
    if st.session_state.df_original is not None:
        df = st.session_state.df_original
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Employees", len(df))
        col2.metric("Features", len(df.columns))
        if 'burnout_risk' in df.columns:
            burnout_count = int(df['burnout_risk'].sum() if pd.api.types.is_numeric_dtype(df['burnout_risk']) else df['burnout_risk'].eq(1).sum())
            col3.metric("High Burnout Risk", f"{burnout_count} ({burnout_count/len(df)*100:.1f}%)")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Instructions")
        st.markdown("""
        1. **Upload Data**: Use the sidebar to upload your CSV file or use the example dataset
        2. **Data Overview**: View basic statistics and missing values
        3. **Data Preprocessing**: Clean and prepare your data for analysis
        4. **EDA**: Explore patterns and correlations in your data
        5. **Model Training**: Train a Random Forest model to predict burnout risk
        6. **Predict**: Use the trained model to predict burnout risk for new employees
        7. **Download Model**: Save your trained model for future use
        """)
    else:
        st.info("👈 Please upload a CSV file or use the example dataset from the sidebar to get started.")

# -----------------------------------
# DATA OVERVIEW
# -----------------------------------
elif page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    if st.session_state.df_original is None:
        st.warning("⚠ Please upload data first from the Home page!")
        st.stop()
    
    df = st.session_state.df_original
    
    tab1, tab2, tab3 = st.tabs(["Dataset", "Info", "Missing Values"])
    
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            st.dataframe(dtype_df)
        with col2:
            st.write("**Basic Statistics:**")
            st.dataframe(df.describe())
    
    with tab3:
        missing = df.isna().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        if missing_df.empty:
            st.success("✅ No missing values!")
        else:
            st.dataframe(missing_df)

# -----------------------------------
# DATA PREPROCESSING
# -----------------------------------
elif page == "🔧 Data Preprocessing":
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.df_original is None:
        st.warning("⚠ Please upload data first!")
        st.stop()
    
    df = st.session_state.df_original.copy()
    
    if st.button("🔄 Start Data Processing"):
        with st.spinner("Processing data..."):
            # Step 1: Handle question marks and missing values (safe coercion)
            question_cols = ['sleep_hours', 'productivity_score', 'peer_feedback',
                             'hours_worked','work_life_balance_score','meetings_count',
                             'remote_days','overtime_hours','stress_level','week','burnout_risk']
            df = replace_question_and_cast(df, question_cols)
            
            # Step 2: Convert zeros to NaN for specified columns
            zero_to_nan_cols = [
                'week', 'hours_worked', 'work_life_balance_score', 'meetings_count',
                'remote_days', 'sleep_hours', 'productivity_score', 'overtime_hours',
                'stress_level', 'peer_feedback'
            ]
            df = set_zeros_to_nan(df, zero_to_nan_cols)
            
            # Step 3: Impute missing values (safe numeric coercion and fallbacks)
            df = impute_columns(df)
            
            # Step 4: Remove outliers
            columns_to_check = ['hours_worked', 'work_life_balance_score', 
                              'meetings_count', 'overtime_hours', 'stress_level']
            df_cleaned, outlier_summary = remove_outliers_iterative(df, columns_to_check)
            
            # If outlier removal removed everything, keep the imputed df and warn
            if df_cleaned.empty:
                st.warning("Outlier removal removed all rows. Reverting to imputed dataset (no outlier filtering).")
                df_cleaned = df.copy()
            
            # Step 5: Drop employee_id if exists
            if 'employee_id' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(['employee_id'], axis=1)
            
            # Store processed data
            st.session_state.df_processed = df_cleaned
            
            st.success("✅ Data processed successfully!")
            
            # Show outlier summary
            st.subheader("Outlier Removal Summary")
            st.json(outlier_summary)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Rows", len(df))
            with col2:
                st.metric("Processed Rows", len(df_cleaned))
    
    if st.session_state.df_processed is not None:
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.df_processed.head(), use_container_width=True)

# -----------------------------------
# EDA
# -----------------------------------
elif page == "📈 EDA":
    st.header("📈 Exploratory Data Analysis")
    
    if st.session_state.df_processed is None:
        st.warning("⚠ Process data first in Data Preprocessing!")
        st.stop()
    
    df = st.session_state.df_processed
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")
    
    # Burnout distribution
    if 'burnout_risk' in df.columns:
        st.subheader("Burnout Risk Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        burnout_counts = df['burnout_risk'].value_counts(dropna=False)
        labels = ['Low Risk', 'High Risk'] if set(burnout_counts.index) <= {0, 1} else burnout_counts.index.astype(str)
        ax1.pie(burnout_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Burnout Risk Distribution")
        
        # Bar chart
        burnout_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Burnout Risk Count')
        ax2.set_xlabel('Burnout Risk')
        ax2.set_ylabel('Count')
        
        st.pyplot(fig)
    
    # Boxplots for key features
    st.subheader("Boxplots of Key Features")
    key_features = ['hours_worked', 'stress_level', 'work_life_balance_score', 'overtime_hours']
    available_features = [f for f in key_features if f in df.columns]
    
    if available_features:
        # create up to 4 subplots dynamically
        n = len(available_features)
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:4]):
            if feature in df.columns:
                df.boxplot(column=feature, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
        # hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.df_processed is None:
        st.warning("⚠ Process data first!")
        st.stop()
    
    df = st.session_state.df_processed.copy()
    
    if 'burnout_risk' not in df.columns:
        st.error("❌ Target column 'burnout_risk' not found in processed data!")
        st.stop()
    
    # Prepare features and target
    X = df.drop(['burnout_risk'], axis=1)
    y = df['burnout_risk']
    
    # Keep only numeric features
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] == 0:
        st.error("No numeric features found!")
        st.stop()
    
    st.session_state.feature_names = X_numeric.columns.tolist()
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
    n_estimators = st.sidebar.slider("n_estimators", 100, 500, 200)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            result = train_and_evaluate(
                X_numeric, y, 
                test_size=test_size, 
                random_state=int(random_state),
                n_estimators=int(n_estimators)
            )
            
            st.session_state.model_result = result
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{result['metrics']['accuracy']:.2%}")
            col2.metric("Precision", f"{result['metrics']['precision']:.2%}")
            col3.metric("Recall", f"{result['metrics']['recall']:.2%}")
            col4.metric("F1 Score", f"{result['metrics']['f1']:.2%}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            st.text(result['classification_report'])

# -----------------------------------
# PREDICT BURNOUT RISK
# -----------------------------------
elif page == "🎯 Predict Burnout Risk":
    st.header("🎯 Predict Burnout Risk")
    
    if st.session_state.model_result is None:
        st.warning("⚠ Train model first!")
        st.stop()
    
    model = st.session_state.model_result['model']
    scaler = st.session_state.model_result['scaler']
    feature_names = st.session_state.feature_names
    
    st.subheader("Enter Employee Details")
    
    col1, col2 = st.columns(2)
    
    # Create input fields for all features
    input_data = {}
    
    with col1:
        for i, feature in enumerate(feature_names[:len(feature_names)//2]):
            if feature == 'hours_worked':
                input_data[feature] = st.slider(f"{feature}", 0, 200, 40)
            elif feature == 'stress_level':
                input_data[feature] = st.slider(f"{feature}", 0, 10, 3)
            elif feature == 'work_life_balance_score':
                input_data[feature] = st.slider(f"{feature}", 0, 10, 6)
            elif feature == 'overtime_hours':
                input_data[feature] = st.slider(f"{feature}", 0, 100, 2)
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    with col2:
        for i, feature in enumerate(feature_names[len(feature_names)//2:]):
            if feature == 'meetings_count':
                input_data[feature] = st.slider(f"{feature}", 0, 50, 3)
            elif feature == 'remote_days':
                input_data[feature] = st.slider(f"{feature}", 0, 7, 2)
            elif feature == 'sleep_hours':
                input_data[feature] = st.slider(f"{feature}", 0, 24, 7)
            elif feature == 'productivity_score':
                input_data[feature] = st.slider(f"{feature}", 0, 100, 75)
            elif feature == 'peer_feedback':
                input_data[feature] = st.slider(f"{feature}", 0, 10, 4)
            elif feature == 'week':
                input_data[feature] = st.slider(f"{feature}", 1, 52, 1)
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("🔮 Predict Burnout Risk"):
        # Prepare input array (ensure correct order)
        input_array = np.array([[float(input_data.get(feature, 0.0)) for feature in feature_names]])
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        # safety: if model doesn't support predict_proba, handle gracefully
        try:
            prediction_proba = model.predict_proba(input_scaled)[0]
        except Exception:
            prediction_proba = None
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"🔥 **High Burnout Risk Detected!**")
            else:
                st.success(f"✅ **Low Burnout Risk**")
            
            st.metric("Predicted Burnout Risk", "HIGH" if prediction == 1 else "LOW")
        
        with col2:
            st.subheader("Prediction Probabilities")
            if prediction_proba is not None:
                # assume binary classes 0 and 1
                prob_df = pd.DataFrame({
                    'Class': ['Low Risk (0)', 'High Risk (1)'],
                    'Probability': [f"{prediction_proba[0]:.2%}", f"{prediction_proba[1]:.2%}"]
                })
                st.dataframe(prob_df, use_container_width=True)
            else:
                st.info("Model does not provide probabilities.")

# -----------------------------------
# DOWNLOAD MODEL
# -----------------------------------
elif page == "💾 Download Model":
    st.header("💾 Download Trained Model")
    
    if st.session_state.model_result is None:
        st.warning("⚠ Train model first!")
        st.stop()
    
    model = st.session_state.model_result['model']
    scaler = st.session_state.model_result['scaler']
    feature_names = st.session_state.feature_names
    
    # Create bundle
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmpf:
        joblib.dump(bundle, tmpf.name)
        tmpf.flush()
        tmpf.seek(0)
        data = tmpf.read()
    
    # Download button
    st.download_button(
        label="📥 Download Model",
        data=data,
        file_name="burnout_prediction_model.joblib",
        mime="application/octet-stream"
    )
    
    st.info("""
    **Model Information:**
    - Model Type: Random Forest Classifier
    - Features: {}
    - Model saved with StandardScaler for preprocessing
    """.format(", ".join(feature_names[:5]) + ("..." if len(feature_names) > 5 else "")))
