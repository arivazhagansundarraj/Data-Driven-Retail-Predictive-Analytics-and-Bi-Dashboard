import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Database Connection Configuration
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "ecommerce_analytics")

def get_db_engine():
    connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_url)

def load_data():
    """Load data from MySQL with fallback to local cleaned CSV file."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            print("Successfully connected to MySQL database. Fetching data...")
            query = "SELECT * FROM online_retail_clean"
            df = pd.read_sql(query, conn)
            # Parse dates explicitly
            df['invoice_date'] = pd.to_datetime(df['invoice_date'])
            print(f"Loaded {df.shape[0]} rows from MySQL.")
            return df
    except Exception as e:
        print(f"Could not load data from MySQL: {e}")
        fallback_csv = "dataset/online_retail_II_cleaned.csv"
        if os.path.exists(fallback_csv):
            print(f"Falling back to loading from local file: {fallback_csv}")
            df = pd.read_csv(fallback_csv, parse_dates=["invoice_date"])
            print(f"Loaded {df.shape[0]} rows from CSV.")
            return df
        else:
            raise FileNotFoundError(f"Neither MySQL database nor fallback CSV '{fallback_csv}' was found.")

def run_rfm_analysis(df):
    """Perform RFM Analysis."""
    print("\n--- Performing RFM Analysis ---")
    # Reference date: 1 day after the latest purchase date in the dataset
    ref_date = df['invoice_date'].max() + pd.Timedelta(days=1)
    print(f"Reference Date for Recency: {ref_date}")
    
    # Calculate R, F, M
    # Recency: Days since last purchase
    # Frequency: Number of unique invoices
    # Monetary: Total sales value
    df['total_spend'] = df['quantity'] * df['unit_price']
    
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (ref_date - x.max()).days,
        'invoice_no': 'nunique',
        'total_spend': 'sum'
    }).reset_index()
    
    rfm.rename(columns={
        'invoice_date': 'recency',
        'invoice_no': 'frequency',
        'total_spend': 'monetary'
    }, inplace=True)
    
    print(f"RFM analysis completed for {rfm.shape[0]} unique customers.")
    return rfm

def run_kmeans_segmentation(rfm):
    """Segment customers using K-Means clustering on normalized RFM metrics."""
    print("\n--- Running K-Means Segmentation ---")
    
    # 1. Handle skewness using log-transform
    rfm_log = rfm[['recency', 'frequency', 'monetary']].copy()
    rfm_log['recency'] = np.log1p(rfm_log['recency'])
    rfm_log['frequency'] = np.log1p(rfm_log['frequency'])
    rfm_log['monetary'] = np.log1p(rfm_log['monetary'])
    
    # 2. Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # 3. Fit K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm['cluster'] = clusters
    
    # 4. Map clusters to meaningful business segments dynamically
    # Calculate average monetary value for each cluster
    cluster_means = rfm.groupby('cluster')['monetary'].mean().sort_values(ascending=False)
    
    # Map clusters based on monetary ranking:
    # Rank 0 (Highest Monetary) -> Champions
    # Rank 1 -> Loyal Customers
    # Rank 2 -> At Risk
    # Rank 3 (Lowest Monetary) -> Hibernating / Lost
    rank_to_label = {
        0: "Champions / VIP",
        1: "Loyal Customers",
        2: "At Risk / Needs Attention",
        3: "Lost / Hibernating"
    }
    
    cluster_mapping = {}
    for rank, cluster_id in enumerate(cluster_means.index):
        cluster_mapping[cluster_id] = rank_to_label[rank]
        
    rfm['rfm_segment'] = rfm['cluster'].map(cluster_mapping)
    print("K-Means customer segmentation complete. Cluster distributions:")
    print(rfm['rfm_segment'].value_counts())
    
    # Drop temp cluster column
    rfm.drop(columns=['cluster'], inplace=True)
    return rfm

def build_clv_features(df, start_date, end_date):
    """Helper to build RFM and tenure features for a customer in a specific date range."""
    # Filter transactional data for the specified date range
    sub_df = df[(df['invoice_date'] >= start_date) & (df['invoice_date'] <= end_date)].copy()
    if sub_df.empty:
        return pd.DataFrame()
        
    sub_df['total_spend'] = sub_df['quantity'] * sub_df['unit_price']
    
    # Reference date is the end date of the period
    ref_date = pd.to_datetime(end_date)
    
    # Extract features
    features = sub_df.groupby('customer_id').agg({
        'invoice_date': [
            lambda x: (ref_date - x.max()).days,  # Recency: days since last purchase
            lambda x: (ref_date - x.min()).days   # Tenure: days since first purchase
        ],
        'invoice_no': 'nunique',                  # Frequency
        'total_spend': ['sum', 'mean']             # Monetary, Avg Order Value
    }).reset_index()
    
    # Flatten columns
    features.columns = ['customer_id', 'recency_days', 'tenure_days', 'frequency', 'monetary', 'avg_order_value']
    return features

def train_clv_model(df):
    """Train a Random Forest Regressor to predict CLV using a Calibration-Holdout split."""
    print("\n--- Training CLV Predictive Model ---")
    
    # Dates for calibration/holdout split
    cal_start = '2009-12-01'
    cal_end = '2011-06-30'
    hold_start = '2011-07-01'
    hold_end = '2011-12-10'
    
    # 1. Build features from calibration period (19 months)
    cal_features = build_clv_features(df, cal_start, cal_end)
    if cal_features.empty:
        print("Warning: Calibration period returned no features. Skipping ML model training.")
        return None
        
    # 2. Build target (future spend) from holdout period (5+ months)
    holdout_df = df[(df['invoice_date'] >= hold_start) & (df['invoice_date'] <= hold_end)].copy()
    holdout_df['total_spend'] = holdout_df['quantity'] * holdout_df['unit_price']
    holdout_spend = holdout_df.groupby('customer_id')['total_spend'].sum().reset_index()
    holdout_spend.rename(columns={'total_spend': 'target_clv'}, inplace=True)
    
    # 3. Merge features and target
    train_data = pd.merge(cal_features, holdout_spend, on='customer_id', how='left')
    train_data['target_clv'] = train_data['target_clv'].fillna(0.0) # No purchase = 0 spend
    
    # Define features and target variable
    X_cols = ['recency_days', 'tenure_days', 'frequency', 'monetary', 'avg_order_value']
    X = train_data[X_cols]
    y = train_data['target_clv']
    
    # 4. Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X, y)
    
    # Evaluate model on training dataset
    y_pred = rf_model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"Random Forest Model trained successfully.")
    print(f"Training R2 Score: {r2:.4f}")
    print(f"Training MAE: {mae:.2f}")
    
    # Return trained model
    return rf_model

def predict_future_clv(df, rfm_df, model):
    """Use the trained model to predict future CLV for all customers."""
    print("\n--- Predicting Future Customer Lifetime Value ---")
    if model is None:
        print("No model available. Setting CLV predictions to total historical spend.")
        rfm_df['predicted_clv'] = rfm_df['monetary']
        return rfm_df
        
    # Build features on the entire dataset to predict the NEXT period value
    end_date = df['invoice_date'].max()
    start_date = df['invoice_date'].min()
    
    full_features = build_clv_features(df, start_date, end_date)
    
    # Map features to align with model inputs
    X_cols = ['recency_days', 'tenure_days', 'frequency', 'monetary', 'avg_order_value']
    X_full = full_features[X_cols]
    
    # Predict future value
    full_features['predicted_clv'] = model.predict(X_full)
    
    # Ensure predicted CLV is not negative
    full_features['predicted_clv'] = full_features['predicted_clv'].clip(lower=0.0)
    
    # Merge back to RFM dataframe
    rfm_df = pd.merge(rfm_df, full_features[['customer_id', 'predicted_clv']], on='customer_id', how='left')
    rfm_df['predicted_clv'] = rfm_df['predicted_clv'].fillna(0.0)
    
    print("CLV prediction completed.")
    return rfm_df

def save_to_mysql(rfm_df):
    """Save customer segments and CLV predictions back to MySQL database."""
    print("\n--- Exporting Predictions and Segments to MySQL ---")
    
    # Reorder columns to match DB schema
    db_cols = ['customer_id', 'recency', 'frequency', 'monetary', 'rfm_segment', 'predicted_clv']
    export_df = rfm_df[db_cols].copy()
    
    # Clean customer_id
    export_df['customer_id'] = export_df['customer_id'].astype(int)
    
    try:
        engine = get_db_engine()
        # Truncate the table first to avoid duplicate primary key errors on rerun
        # and to keep table clean.
        with engine.begin() as conn:
            print("Truncating existing 'customer_segments_clv' table data...")
            conn.execute(text("TRUNCATE TABLE customer_segments_clv"))
            
        print("Inserting records...")
        export_df.to_sql(
            name="customer_segments_clv",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=5000
        )
        print("Successfully exported segments and predictions to MySQL database!")
    except Exception as e:
        print(f"Error exporting data to MySQL: {e}")
        # Save output to a local CSV in case DB is unreachable
        local_output = "dataset/customer_segments_clv_output.csv"
        export_df.to_csv(local_output, index=False)
        print(f"Saved results locally as backup: {local_output}")

def main():
    # 1. Load Cleaned Transactions
    df = load_data()
    
    # 2. Run RFM Analysis
    rfm = run_rfm_analysis(df)
    
    # 3. K-Means Clustering
    rfm = run_kmeans_segmentation(rfm)
    
    # 4. Train CLV Model
    clv_model = train_clv_model(df)
    
    # 5. Predict CLV
    rfm = predict_future_clv(df, rfm, clv_model)
    
    # 6. Save back to MySQL
    save_to_mysql(rfm)
    
    print("\nMachine Learning Pipeline executed successfully!")

if __name__ == "__main__":
    main()
