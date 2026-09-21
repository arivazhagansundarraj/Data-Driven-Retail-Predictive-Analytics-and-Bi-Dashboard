import os
import pandas as pd
from sqlalchemy import create_engine

# Database Connection Configuration
# Modify these variables to match your MySQL database configuration or set environment variables.
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "ecommerce_analytics")
TABLE_NAME = "online_retail_clean"

def load_data_to_mysql(cleaned_csv_path):
    print("Starting database insertion pipeline...")
    
    # 1. Check if cleaned CSV exists
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned data file not found at: {cleaned_csv_path}. Please run clean_data.py first.")
    
    # 2. Build SQLAlchemy Engine
    # Using mysql+pymysql driver
    connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Connecting to database '{DB_NAME}' on '{DB_HOST}'...")
    
    try:
        engine = create_engine(connection_url)
        # Test connection
        with engine.connect() as conn:
            print("Successfully connected to MySQL database!")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please ensure your MySQL server is running, the database exists, and your credentials are correct.")
        return

    # 3. Read cleaned data
    print(f"Reading cleaned dataset from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path, parse_dates=["invoice_date"])
    print(f"Loaded {df.shape[0]} rows ready for insertion.")

    # 4. Insert data using df.to_sql in chunks
    print(f"Inserting records into table '{TABLE_NAME}'...")
    try:
        # We append because the schema.sql already creates the table structure.
        # index=False ensures the DataFrame index is not written as a column.
        # chunksize=10000 ensures efficient bulk loading without memory issues.
        df.to_sql(
            name=TABLE_NAME, 
            con=engine, 
            if_exists="append", 
            index=False, 
            chunksize=10000
        )
        print("Data loaded to MySQL database successfully!")
    except Exception as e:
        print(f"Failed to load data into database: {e}")

if __name__ == "__main__":
    CLEANED_CSV = "dataset/online_retail_II_cleaned.csv"
    load_data_to_mysql(CLEANED_CSV)
