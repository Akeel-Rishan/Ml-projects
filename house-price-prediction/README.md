# House Price Prediction

A machine learning project for predicting house prices using various regression models.

## Project Structure

```
house-price-prediction/
├── data/                    # Raw and processed data
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessor.py     # Data preprocessing
│   ├── models.py           # Model training functions
│   └── evaluator.py        # Model evaluation metrics
├── outputs/                # Model outputs and results
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main pipeline:
```bash
python main.py
```

Or explore the data interactively:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Models

The project includes implementations for:
- Linear Regression
- Random Forest
- Gradient Boosting

## License

MIT
