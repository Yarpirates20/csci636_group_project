## Preprocessing Report
The purpose of this report is to document data sources, how it was cleaned, what features were created, and how datasets were prepared for model training.

### Data Sources and Collection
| Dataset | Source | Variables |
| --- | --- | --- |
| Historical Bitcoin Prices | Steven's website | Open, high, low, close |
| Bitcoin on-chain data | Coin Metrics | Active addresses, MVRV, transaction fees ,hash rate, transaction count|
---

##### Additional Info
- Date Range: 2018 - May 23, 2026
- Frequency: Daily
- Date format: UTC
- Purpose: compare price-only prediction with price + on-chain prediction.

### Data Cleaning Procedures

- **Date standardization**: Converted date strings to Pandas datetime objects using UTC

- **Missing values**: Price data had no missing values. On-chain data had missing observations initially. After restricting on-chain dataset to historical preparation period starting December 1, 2017, only one missing observation remained per metric, which was removed. The cleaned on-chain dataset contained 3096 observations, including 31 observations from December 2017 used for feature engineering. The modeling datasets were restricted to January 1, 2018-May 23, 2026 which resulted in 3065 observations each.

- **Duplicate detection**: Used Pandas 'duplicated()' method to identify duplicates based on date and 'drop_duplicates()' method to drop them.

- **Invalid values**: 
    - **Price Dataset**: The price dataset was checked to see if any of the columns (open, low, high, close) were less than or equal to zero, if 'high' was less than any of the other columns, or if 'low' was higher than the other columns. Summary statistics were then checked to look for outliers.

    - **On-chain dataset**: The Coin Metrics dataset was examined for invalid observations such as negative active address counts, transaction counts, transaction fees, and hash rates, as well as nonpositive MVRV values. Active addresses and transaction counts were checked for non-integer values. No violations or infinite values were found. Descriptive statistics were calculated for all five on-chain metrics, each contained 3096 observations. There was variation in transaction fees and hash rate, which seems consistent with Bitcoin network activity over time. Since extreme values do not necessarily mean there were errors, these values were retained.

- **Outliers**: Legitimate extreme price/on-chain metrics were retained since prices can be volatile. 

- **Frequency**: Both sets were standardized to daily observations by reindexing datasets to a complete daily calendar -- shifting by one row means shifting by one day. Standardization occurred before feature engineering and merging.


### Feature Engineering
After cleaning the datasets, additional features were calculated to identify historical patterns that could be useful for predicting next-day price direction. The baseline dataset has 14 price-based features, and the enhanced dataset includes those plus 9 additional on-chain features.

**Price-Based Features**
Price based features include: BTC daily opening, highest, lowest, and closing (OHLC) prices. Daily, 7-day, and 30-day returns calculated to measure percentage changes in closing prices over different periods. 7-day and 30-day moving averages calculated to identify short-term and longer-term trends. Rolling volatility over these periods (std. dev.) used to measure variation in daily returns. 1-day and 2-day lagged returns included to capture previous price movements. Normalized daily price range calculated by dividing difference between the daily high and low prices by the opening price.

**On-Chain Features**
Enhanced dataset has 5 metrics: active addresses, MVRV ratio, transaction fees, hash rate, and transaction count. These features provide additional info about BTC network activity, valuation, and mining conditions. 7-day percentage changes calculated for active addresses, transaction fees, hash rate, and transaction count to capture network activity changes over time.

Engineered features were calculated using historical observations rather than future prices/network activity to reduce data leakage and ensure models did not use future information unavailable at prediction time.

All on-chain features were shifted forward by one day so that each prediction used the previous day's network metrics. This was done to account for potential reporting delays and reduce look-ahead bias..

**Target Variable Creation**
Predictions were designed to be made at the end of each trading day, using that day's completed price information and previously available on-chain metrics to predict the following day's closing price direction.

A binary target variable was created to represent the next-day price direction for Bitcoin. If the next day's closing price was higher than the current day's closing price, the target was assigned a value of 1. Otherwise, it was assigned a value of 0. The next-day return was exluded from the model's input features to prevent data leakage since it contains information about the future price movement being predicted. 

**Dataset Construction**
After feature engineering, the price and on-chain datasets were merged using their standardized dates. Two datasets were constructed for comparison. The baseline dataset has 14 price-based features, and the enhanced has those along with 9 additional on-chain features. Both datasets contain 3065 observations, with 15 and 24 columns, including the target variable. Observations with missing values were exluded, and both datasets were restricted to identical dates to ensure fair comparison.

**Training, Validation, and Testing Splits**
Since we were working with time-series data, the data was split chronologically into training, validation, and testing sets. Data from 2018-2024 was used for training, 2025 for validation, and January-May 23, 2026 was used for testing. The same date range was used for baseline and enhanced datasets. 

December 31, for 2024 and 2025 were excluded to prevent next-day target labels from crossing boundaries between datasets. 

**Summary**
The preprocessing produced two cleaned source datasets and two feature-engineered datasets for model comparison. The baseline contains only historical price information, while the enhanced has additional on-chain metrics. Each dataset was divided into training, validation, and testing, resulting in six CSV files for modeling. The collection, cleaning, and feature engineering were documented in separate Jupyter notebooks. 