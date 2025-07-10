# Dataset Information

## Credit Card Fraud Detection Dataset

**Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Dataset Description
- **Total Transactions**: 284,807
- **Fraud Cases**: 492 (0.173%)
- **Time Period**: 2 days
- **Features**: 30 total
  - Time: Number of seconds elapsed between each transaction and the first transaction
  - V1-V28: PCA-transformed features (anonymized)
  - Amount: Transaction amount
  - Class: 0 = Normal, 1 = Fraud

### Download Instructions
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place in this `data/` directory

### Data Privacy
- Customer identities and sensitive features are anonymized using PCA
- Only transaction patterns and amounts are preserved
- Suitable for research and educational purposes

### File Structure
```
data/
├── README.md          # This file
└── creditcard.csv     # Main dataset (download from Kaggle)
```