-- MySQL Schema for E-Commerce Customer Retention and Value Optimization System
-- Phase 1: Data Engineering Pipeline

CREATE DATABASE IF NOT EXISTS ecommerce_analytics;
USE ecommerce_analytics;

-- Table to store cleaned transactions
CREATE TABLE IF NOT EXISTS online_retail_clean (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_no VARCHAR(20) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    description VARCHAR(255),
    quantity INT NOT NULL,
    invoice_date DATETIME NOT NULL,
    unit_price DECIMAL(10, 4) NOT NULL,
    customer_id INT NOT NULL,
    country VARCHAR(100),
    
    -- Indexes for optimizing RFM & CLV calculations in Phase 2
    INDEX idx_customer_id (customer_id),
    INDEX idx_invoice_date (invoice_date)
);

-- Table to store ML customer segmentation and CLV predictions
CREATE TABLE IF NOT EXISTS customer_segments_clv (
    customer_id INT PRIMARY KEY,
    recency INT NOT NULL,
    frequency INT NOT NULL,
    monetary DECIMAL(12, 2) NOT NULL,
    rfm_segment VARCHAR(50) NOT NULL,
    predicted_clv DECIMAL(12, 2) NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

