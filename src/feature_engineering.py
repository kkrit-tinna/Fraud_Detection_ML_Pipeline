# Enhanced Credit Card Fraud Detection - Feature Engineering Module
# Streamlined ML Pipeline with Feature Engineering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import datetime as dt

# Set display options
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 50)

def load_and_prepare_data(train_file_path='../../data/fraudTrain.csv', test_file_path='../../data/fraudTest.csv'):
    """Load and perform initial data preparation"""
    # Load datasets
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    
    # Rename unnamed column
    train_data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    test_data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Fraud rate in training: {train_data['is_fraud'].mean():.3%}")
    print(f"Fraud rate in test: {test_data['is_fraud'].mean():.3%}")
    
    return train_data, test_data

def comprehensive_data_quality_check(df, name):
    """Enhanced data quality assessment"""
    print(f"\n=== {name} Data Quality Report ===")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nMissing values found:")
        print(missing_data[missing_data > 0])
    else:
        print("No missing values")
    
    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Target variable analysis
    if 'is_fraud' in df.columns:
        fraud_rate = df['is_fraud'].mean()
        print(f"\nFraud rate: {fraud_rate:.3%}")
        print(f"Class imbalance ratio: {(1-fraud_rate)/fraud_rate:.1f}:1")
    
    return {
        'missing_values': missing_data.sum(),
        'duplicates': duplicates,
        'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else None
    }

def extract_temporal_features(df):
    """Enhanced temporal feature extraction"""
    df = df.copy()
    
    # Convert datetime columns
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    # Extract basic temporal features
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    df['trans_day'] = df['trans_date_trans_time'].dt.day_name()
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_weekday'] = df['trans_date_trans_time'].dt.weekday
    
    # Advanced temporal features
    df['is_weekend'] = df['trans_weekday'].isin([5, 6]).astype(int)
    df['is_night'] = df['trans_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['is_business_hours'] = df['trans_hour'].isin(range(9, 17)).astype(int)
    
    # Seasonal features
    df['quarter'] = df['trans_date_trans_time'].dt.quarter
    df['is_holiday_season'] = df['trans_month'].isin([11, 12]).astype(int)
    df['is_tax_season'] = df['trans_month'].isin([1, 2, 3, 4]).astype(int)
    
    # Time-based cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['trans_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['trans_hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['trans_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['trans_month'] / 12)
    
    return df

def extract_demographic_features(df):
    """Enhanced demographic feature extraction"""
    df = df.copy()
    
    # Calculate age
    ref_date = pd.to_datetime('2021-01-01')
    df['age'] = (ref_date - df['dob']).dt.days // 365
    
    # Age groups with more granularity
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 18, 25, 35, 50, 65, 100], 
                           labels=['minor', 'young_adult', 'adult', 'middle_aged', 'senior', 'elderly'])
    
    # Gender encoding
    df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
    
    return df

def extract_geographic_features(df):
    """Enhanced geographic feature extraction"""
    df = df.copy()
    
    # City population categories
    df['city_pop_category'] = pd.cut(df['city_pop'],
                                   bins=[0, 10000, 50000, 100000, 500000, 1000000, np.inf],
                                   labels=['rural', 'small_town', 'town', 'small_city', 'city', 'major_city'])
    
    # Distance between cardholder and merchant
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['is_distant_transaction'] = (df['distance_km'] > 100).astype(int)
    
    return df

def prepare_model_data(train_df, test_df):
    """Advanced data preprocessing pipeline"""
    
    # Define columns to drop
    drop_cols = [
        'id', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 
        'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 
        'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'trans_day', 'distance_km', 'quarter', 'trans_weekday'
    ]
    
    # Drop columns that exist
    existing_drop_cols = [col for col in drop_cols if col in train_df.columns]
    
    train_processed = train_df.drop(existing_drop_cols, axis=1, errors='ignore')
    test_processed = test_df.drop(existing_drop_cols, axis=1, errors='ignore')
    
    # Handle categorical variables
    categorical_cols = train_processed.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != 'is_fraud']
    
    # Label encoding
    for col in categorical_cols:
        le = LabelEncoder()
        # Combine train and test to ensure consistent encoding
        combined = pd.concat([train_processed[col], test_processed[col]], axis=0)
        le.fit(combined.dropna())
        train_processed[col] = le.transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
    
    # Align columns between train and test
    missing_cols_test = set(train_processed.columns) - set(test_processed.columns)
    for col in missing_cols_test:
        if col != 'is_fraud':
            test_processed[col] = 0
    
    missing_cols_train = set(test_processed.columns) - set(train_processed.columns)
    for col in missing_cols_train:
        if col != 'is_fraud':
            train_processed[col] = 0
    
    # Reorder columns
    feature_cols = [col for col in train_processed.columns if col != 'is_fraud']
    if 'is_fraud' in train_processed.columns:
        train_processed = train_processed[feature_cols + ['is_fraud']]
    test_processed = test_processed[feature_cols + (['is_fraud'] if 'is_fraud' in test_processed.columns else [])]
    
    return train_processed, test_processed

def apply_smote_balancing(X_train, y_train):
    """Apply SMOTE for class balancing"""
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced dataset shape: {X_train_balanced.shape}")
    print(f"Fraud rate after SMOTE: {y_train_balanced.mean():.3%}")
    return X_train_balanced, y_train_balanced

def create_fraud_analysis_plots(train_enhanced):
    """Create comprehensive fraud analysis visualizations"""
    fraud_data = train_enhanced[train_enhanced['is_fraud'] == 1]
    normal_data = train_enhanced[train_enhanced['is_fraud'] == 0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Amount distribution
    axes[0,0].hist(normal_data['amt'], bins=50, alpha=0.7, label='Normal', density=True)
    axes[0,0].hist(fraud_data['amt'], bins=50, alpha=0.7, label='Fraud', density=True)
    axes[0,0].set_xlabel('Transaction Amount')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Transaction Amount Distribution')
    axes[0,0].legend()
    axes[0,0].set_xlim(0, 1000)
    
    # Hourly patterns
    hour_fraud = fraud_data['trans_hour'].value_counts().sort_index()
    hour_normal = normal_data['trans_hour'].value_counts().sort_index()
    hour_fraud_rate = hour_fraud / (hour_fraud + hour_normal)
    
    sns.barplot(x=hour_fraud_rate.index, y=hour_fraud_rate.values, ax=axes[0,1], palette='rocket_r')
    axes[0,1].set_xlabel('Hour of Day')
    axes[0,1].set_ylabel('Fraud Rate')
    axes[0,1].set_title('Fraud Rate by Hour')
    
    # Age group analysis
    age_fraud = pd.crosstab(train_enhanced['age_group'], train_enhanced['is_fraud'], normalize='index')
    sns.barplot(x=age_fraud.index, y=age_fraud[1].values, ax=axes[0,2], palette='Paired')
    axes[0,2].set_xlabel('Age Group')
    axes[0,2].set_ylabel('Fraud Rate')
    axes[0,2].set_title('Fraud Rate by Age Group')
    
    # Distance analysis
    axes[1,0].hist(normal_data['distance_km'], bins=50, alpha=0.7, label='Normal', density=True)
    axes[1,0].hist(fraud_data['distance_km'], bins=50, alpha=0.7, label='Fraud', density=True)
    axes[1,0].set_xlabel('Distance (km)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Distance Distribution')
    axes[1,0].legend()
    axes[1,0].set_xlim(0, 500)
    
    # Monthly distribution
    month_fraud = fraud_data['trans_month'].value_counts().sort_index()
    month_normal = normal_data['trans_month'].value_counts().sort_index()
    month_fraud_rate = month_fraud / (month_fraud + month_normal)
    
    sns.barplot(x=month_fraud_rate.index, y=month_fraud_rate.values, ax=axes[1,1], palette='viridis')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Fraud Rate')
    axes[1,1].set_title('Fraud Rate by Month')
    
    # Gender distribution
    gender_fraud = pd.crosstab(train_enhanced['gender'], train_enhanced['is_fraud'], normalize='index')
    sns.barplot(x=gender_fraud.index, y=gender_fraud[1].values, ax=axes[1,2], palette='Set2')
    axes[1,2].set_xlabel('Gender')
    axes[1,2].set_ylabel('Fraud Rate')
    axes[1,2].set_title('Fraud Rate by Gender')
    
    plt.tight_layout()
    plt.savefig("fraud_analysis_plots.png")
    plt.show()

def full_feature_engineering_pipeline(train_file_path='../../data/fraudTrain.csv', test_file_path='../../data/fraudTest.csv'):
    """Complete feature engineering pipeline"""
    print("Starting feature engineering pipeline...")
    
    # Load data
    train_data, test_data = load_and_prepare_data(train_file_path, test_file_path)
    
    # Quality checks
    train_quality = comprehensive_data_quality_check(train_data, "Training")
    test_quality = comprehensive_data_quality_check(test_data, "Test")
    
    # Apply feature engineering
    print("\nApplying enhanced feature engineering...")
    train_data_enhanced = extract_temporal_features(train_data)
    train_data_enhanced = extract_demographic_features(train_data_enhanced)
    train_data_enhanced = extract_geographic_features(train_data_enhanced)
    
    test_data_enhanced = extract_temporal_features(test_data)
    test_data_enhanced = extract_demographic_features(test_data_enhanced)
    test_data_enhanced = extract_geographic_features(test_data_enhanced)
    
    print(f"Enhanced training data shape: {train_data_enhanced.shape}")
    print(f"Enhanced test data shape: {test_data_enhanced.shape}")
    
    # Prepare model data
    train_processed, test_processed = prepare_model_data(train_data_enhanced, test_data_enhanced)
    
    # Split features and target
    X_train = train_processed.drop('is_fraud', axis=1)
    y_train = train_processed['is_fraud']
    
    if 'is_fraud' in test_processed.columns:
        X_test = test_processed.drop('is_fraud', axis=1)
        y_test = test_processed['is_fraud']
    else:
        X_test = test_processed
        y_test = None
    
    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote_balancing(X_train, y_train)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': X_train.columns,
        'train_enhanced': train_data_enhanced,
        'test_enhanced': test_data_enhanced
    }