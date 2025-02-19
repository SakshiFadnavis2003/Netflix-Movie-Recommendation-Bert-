# 🎬 Movie Recommendation System using BERT

## 📌 Overview
This project is a **Movie Recommendation System** that leverages **BERT embeddings** and **cosine similarity** to suggest similar movies based on their titles, genres, and descriptions. It utilizes **Transformers (Hugging Face), PyTorch, and Scikit-learn** to compute semantic similarities between movies.

## 📂 Dataset
- The dataset used is `netflix_titles.csv`, which contains information about movies and TV shows on Netflix, including **title, description, genre, and more**.
- The dataset is preprocessed by concatenating relevant text fields to form meaningful content for similarity calculations.

## 🚀 Features
- **BERT for text embeddings**: Converts movie descriptions into dense vector representations.
- **Cosine Similarity**: Measures the semantic similarity between movies.
- **Efficient Recommendations**: Given a movie title, retrieves the **top 10 most similar movies**.

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/SakshiFadnavis2003/Movie-Recommendation-System-BERT.git
cd Movie-Recommendation-System-BERT
```
### 2️⃣ Install Dependencies
Ensure you have Python **3.8+** and install the required libraries:
```sh
pip install -r requirements.txt
```
(If `requirements.txt` is not available, manually install:)
```sh
pip install pandas torch transformers scikit-learn numpy matplotlib
```
### 3️⃣ Run the System
```sh
python simple-matplotlib-visualization-tips.ipynb
```

## 📜 How It Works
1. **Load the dataset** and preprocess movie titles, genres, and descriptions.
2. **Generate embeddings** for each movie using **BERT**.
3. **Compute pairwise cosine similarity** to find similar movies.
4. **Retrieve recommendations** for a given movie title.

## 📌 Example Usage
```python
from recommend import get_recommendations

# Example: Get recommendations for a specific movie
title = "Inception"
recommendations = get_recommendations(title, df, cosine_sim)
print(recommendations)
```
**🔹 Output:**
```
1. Legion
2. Lady in the Water
3. Prospect
4. Lockout
5. Mute
6. The Spy Next Door
7. Underworld
8. Incoming
9. White Chamber
10. Jupiter Ascending
```

## 📊 XKCD Plot Example
This project also includes an **XKCD-style** visualization using `matplotlib`:
```python
import matplotlib.pyplot as plt
import numpy as np

with plt.xkcd():
    fig, ax = plt.subplots()
    data = np.ones(100)
    data[70:] -= np.arange(30)
    ax.plot(data)
    ax.set_title("Stove Ownership from XKCD")
    plt.show()
```

## 📜 File Structure
```
📁 Movie-Recommendation-System-BERT
│── 📄 simple-matplotlib-visualization-tips.ipynb  # Main script for recommendations
│── 📄 netflix_titles.csv  # Dataset
│── 📄 requirements.txt  # Dependencies
│── 📄 README.md  # Project documentation
```

## 🏆 Future Improvements
- Implement **FAISS or Annoy** for **faster similarity search**.
- Add **Graph-based approaches** to enhance recommendations.
- Build a **web app** using **Flask or Streamlit** for easy user interaction.

## 📬 Contact
💡 **Author:** [Sakshi Fadnavis](https://github.com/SakshiFadnavis2003)  
📩 **Email:** fadnavissakshi@gmail.com  
🔗 **LinkedIn:** [Sakshi Fadnavis](https://www.linkedin.com/in/sakshi-fadnavis-3023a9240/)  

---
🌟 **If you like this project, give it a ⭐ on GitHub!** 🚀

