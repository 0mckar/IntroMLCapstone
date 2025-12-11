# IntroMLCapstone

## 🚀 Installation & Setup

### Option 1: Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#  Using Conda
# Create and activate conda environment
conda env create -f environment.yml
conda activate house-price-prediction

# Dependencies
python==3.9+
pandas==1.5.0
numpy==1.24.0
scikit-learn==1.2.0
xgboost==1.7.0
matplotlib==3.6.0
seaborn==0.12.0
jupyter==1.0.0
notebook==6.5.0
torch==2.0.0  # For neural network implementation

run:
jupyter notebook notebooks/main.ipynb

