# 🐦 Twitter Sentiment Analysis

A Streamlit web application for analyzing text sentiment using a pre-trained LSTM neural network model.

## 🚀 Features

- **4-Class Sentiment Classification**: Negative, Positive, Neutral, and Irrelevant
- **Real-time Analysis**: Get instant sentiment predictions for any text input
- **Beautiful UI**: Modern, responsive design with custom styling
- **Model Accuracy**: 85% test accuracy

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Quick Start
```bash
# Clone the repository
git clone <your-repo-url>
cd Twitter-Sentiment-Analysis

# Activate virtual environment
venv_senti\Scripts\activate  # Windows
# source venv_senti/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 How to Use

1. **Input Text**: Enter any text in the text area
2. **Analyze**: Click the "Predict" button
3. **View Results**: See the predicted sentiment and confidence scores

## 🎯 Sentiment Classes

| Class | Description |
|-------|-------------|
| **Positive** | Expresses positive emotions, satisfaction, or approval |
| **Negative** | Expresses negative emotions, dissatisfaction, or disapproval |
| **Neutral** | Factual content without strong emotional content |
| **Irrelevant** | Off-topic or unclear sentiment |

## 📁 Project Structure

```
Twitter Sentiment Analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── Sentiment_Analysis_85_test_Acc.h5  # Pre-trained model
├── tokenizer.pkl                  # Trained tokenizer
└── notebook/                      # Jupyter notebook with model training
    └── twitter-sentiment-analysis-lstm.ipynb
```

## 🔧 Dependencies

- **TensorFlow**: Deep learning framework
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing
- **NumPy**: Numerical computing

## 📚 Model Details

- **Architecture**: LSTM with Embedding layer
- **Input**: Text sequences (max 100 tokens)
- **Output**: 4 sentiment classes
- **Accuracy**: 85% on test data


