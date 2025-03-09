import streamlit as st
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Set Streamlit theme colors
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #181818;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput, .stTextArea, .stButton, .stSelectbox, .stSidebar, .stDataFrame, .stAlert {
        border-radius: 10px;
    }
    .stSidebar {
        background-color: #2c2f33;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
    }
    .stDataFrame {
        border: 2px solid #ff7043;
    }
    .stAlert {
        font-size: 18px;
        font-weight: bold;
    }
    .custom-box {
        padding: 3px 10px;
        background-color: #ff7043;
        color: black;
        border-radius: 6px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        width: fit-content;
        margin: auto;
    }
    .footer {
        font-size: 10px;
        text-align: center;
        color: #aaa;
    }
    .sidebar-box {
        padding: 6px;
        background-color: #4caf50;
        color: white;
        border-radius: 8px;
        text-align: center;
        font-size: 11px;
        font-weight: bold;
        margin-top: 20px;
    }
    .title-text {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        color: black;
    }
    .input-label {
        position: relative;
        left: 0;
        top: 15px;
        font-size: 14px;
        color: #ff7043;
    }
    .tradeoff-box {
        padding: 10px;
        background-color: #333;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
    }
    .tradeoff-table {
        width: 100%;
        border-collapse: collapse;
    }
    .tradeoff-table th, .tradeoff-table td {
        border: 1px solid #ff7043;
        padding: 8px;
        text-align: center;
        color: white;
    }
    .tradeoff-table th {
        background-color: #ff7043;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dataset selection
st.sidebar.title("📂 Select Dataset")
dataset_option = st.sidebar.radio("Choose dataset", [
    "train_1k.csv", "train_3k.csv", "train_5k.csv", "train_8k.csv", "train_20k.csv"
])

# Load selected dataset
news_df = pd.read_csv(dataset_option)
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)

# Calculate accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred) * 100

# Sidebar Visualization Options
st.sidebar.subheader("📊 Visualization")
data_option = st.sidebar.selectbox("🔎 Select Visualization", ["Dataset Overview", "Label Distribution"])
if data_option == "Dataset Overview":
    st.sidebar.dataframe(news_df[['author', 'title', 'label']].head(10))
elif data_option == "Label Distribution":
    fig, ax = plt.subplots()
    news_df['label'].value_counts().plot(kind='bar', ax=ax, color=['blue', 'red'])
    ax.set_title("Label Distribution (0: Real, 1: Fake)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    st.sidebar.pyplot(fig)

st.sidebar.markdown(
    """
    <div class='tradeoff-box'>
    <table class='tradeoff-table'>
        <tr><th>Dataset</th><th>Accuracy</th></tr>
        <tr><td>1K</td><td>93.50%</td></tr>
        <tr><td>3K</td><td>94.67%</td></tr>
        <tr><td>5K</td><td>95.70%</td></tr>
        <tr><td>8K</td><td>95.94%</td></tr>
        <tr><td>20K</td><td>97.91%</td></tr>
    </table>
    </div>
    """,
    unsafe_allow_html=True
)

st.image("banner.png", use_container_width=True)
st.markdown("<h2 class='title-text'>📰 Fake News Detector</h2>", unsafe_allow_html=True)
st.markdown(f"<div class='custom-box'>Model Accuracy: <b>{accuracy:.2f}%</b></div>", unsafe_allow_html=True)

st.markdown("<div class='input-label'>✍️ Enter a news article:</div>", unsafe_allow_html=True)
input_text = st.text_area("", height=70)

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]
st.markdown(
    """
    <style>
        .stButton > button {
            background-color: #ff7043;
            color: black;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
            transition: all 0.3s ease-in-out;
            width: 100%;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }

        .stButton > button:hover {
            background-color: #ff5722;
            color: white;
            transform: scale(1.05);
            box-shadow: 4px 4px 15px rgba(255, 87, 34, 0.6);
        }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔍 Check News", help="Click to analyze the news article"):
        if input_text:
            pred = prediction(input_text)
            if pred == 1:
                st.error("⚠️ The News is Fake!")
            else:
                st.success("✅ The News is Real!")
        else:
            st.warning("⚠️ Please enter a news article.")

import streamlit as st

# Custom CSS for sidebar styling to match theme
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #0A192F !important; /* Dark Navy */
            padding: 20px;
            border-right: 2px solid #1E90FF;
        }
        [data-testid="stSidebar"] h2 {
            color: #F8F9FA; /* Light Grey-White */
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        [data-testid="stSidebar"] a {
            color: #1E90FF; /* Bright Blue */
            text-decoration: none;
            font-weight: bold;
            transition: 0.3s;
        }
        [data-testid="stSidebar"] a:hover {
            color: #FFD700; /* Gold */
            text-shadow: 0px 0px 10px #FFD700;
        }
        [data-testid="stSidebar"] p {
            color: #F8F9FA;
            font-size: 16px;
            margin: 10px 0;
        }
        [data-testid="stSidebar"] hr {
            border: 1px solid #1E90FF;
            margin: 15px 0;
        }
        [data-testid="stSidebar"] .icon {
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🖥️ GitHub Link")
st.sidebar.markdown("🔗 [View on GitHub](https://github.com/Masuddar/Hackoona-Matata-2025?tab=readme-ov-file)")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎥 Video Tutorial")
st.sidebar.markdown("📺 **Watch Here:** [YouTube Tutorial](https://www.youtube.com/)")  # Replace with actual link

st.sidebar.markdown("---")
st.sidebar.markdown("## 🤝 Connect & Support")
st.sidebar.markdown("📧 **Email:** masuddarrahaman31@gmail.com  \n"
                    "🌐 **My Website:** [Masuddar.in](https://masuddar.in/)  \n"
                    "📂 **Portfolio:**  [My Portfolio](https://masuddar.netlify.app/)  \n"
                    "🔗 **LinkedIn:**  [My LinkedIn Profile](https://www.linkedin.com/in/masuddar-rahaman-b5044b283/)")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📞 Contact the Organisers")
st.sidebar.markdown("🏛 **IIIT Kottayam - BetaLabs**  \n"
                    "📩 **Email:** techclub@iiitkottayam.ac.in  \n"
                    "📱 **Phone:** +91 9100862186")


st.markdown("---")
st.markdown("<div class='footer'>By Masuddar Rahaman</div>", unsafe_allow_html=True)
