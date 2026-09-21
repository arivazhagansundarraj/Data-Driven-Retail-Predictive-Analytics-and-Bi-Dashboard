import os
import pandas as pd

def clean_ecom_data(raw_filepath, clean_filepath):
    print("Starting data cleaning pipeline...")
    
    # 1. Check if the raw file exists
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Raw data file not found at: {raw_filepath}")
        
    print(f"Loading raw data from: {raw_filepath}")
    # Read the CSV file
    df = pd.read_csv(raw_filepath)
    print(f"Raw data shape: {df.shape}")
    
    # 2. Drop rows where Customer ID is missing
    print("Dropping rows with missing Customer ID...")
    df = df.dropna(subset=['Customer ID'])
    print(f"Shape after dropping missing Customer IDs: {df.shape}")
    
    # 3. Convert Customer ID to integer (it loads as float due to NaN support in older pandas)
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # 4. Remove cancellations (Invoices starting with 'C')
    # Convert Invoice column to string to ensure string matching works
    df['Invoice'] = df['Invoice'].astype(str).str.strip()
    print("Removing cancellations...")
    df = df[~df['Invoice'].str.startswith('C', na=False)]
    print(f"Shape after removing cancellations: {df.shape}")
    
    # 5. Filter for positive quantities and unit prices
    print("Filtering negative and zero quantities and unit prices...")
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    print(f"Shape after filtering positive values: {df.shape}")
    
    # 6. Format columns
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Rename columns to match MySQL database schema
    df = df.rename(columns={
        'Invoice': 'invoice_no',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'Price': 'unit_price',
        'Customer ID': 'customer_id',
        'Country': 'country'
    })
    
    # Select and order columns matching database schema
    db_cols = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'unit_price', 'customer_id', 'country']
    df = df[db_cols]
    
    # 7. Save cleaned data to CSV
    print(f"Saving cleaned dataset to: {clean_filepath}")
    os.makedirs(os.path.dirname(clean_filepath), exist_ok=True)
    df.to_csv(clean_filepath, index=False)
    print("Data cleaning completed successfully!")
    print(f"Cleaned dataset shape: {df.shape}")

if __name__ == "__main__":
    RAW_PATH = "dataset/online_retail_II.csv"
    CLEAN_PATH = "dataset/online_retail_II_cleaned.csv"
    clean_ecom_data(RAW_PATH, CLEAN_PATH)
