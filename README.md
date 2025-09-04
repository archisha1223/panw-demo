youtube link: https://youtu.be/Tvny1eAPlMI

# 💸 Smart Financial Coach

This project is a personal finance web app that mimics the guidance of a real financial coach — using AI to analyze your transactions and generate insights, forecasts, and warnings. It was built for the PANW case challenge.

### ⚡ Features (3 MVPs)
1. **AI-Powered Spending Insights**:  
   Detects trends, anomalies, and spending patterns using Theil-Sen Regression, Isolation Forests, and heuristics.

2. **Personalized Goal Forecasting**:  
   Forecasts whether you’ll meet a savings goal based on predicted spend (via per-category Linear Regression) and income.

3. **Subscriptions & Gray Charges Detector**:  
   Unsupervised-learning heuristics (based on cadence + variation) flag recurring charges, free trial conversions, and microtransactions.

---

### 🧠 Tech Stack
- **Backend**: FastAPI  
- **Frontend**: Jinja2 templates + Chart.js  
- **AI Models**: scikit-learn (Theil–Sen, Isolation Forest, Linear Regression)  
- **Data Wrangling**: pandas, NumPy  
- **Deployment**: Localhost (for demo)

---

### ▶️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/archisha1223/panw-demo.git
cd panw-demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app.main:app --reload


Then open your browser to:
http://127.0.0.1:8000
