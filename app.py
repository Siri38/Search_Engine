import joblib
import pandas as pd
import os
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("final.csv")
count_vectorizer = joblib.load("count_vectorizer.joblib")
tfidf_transformer = joblib.load("tfidf_transformer.joblib")
tfidf_matrix = joblib.load("tfidf_matrix.joblib")


def generate_summarized_documents(retrieved_documents):
    summarized_documents = {}
    for i, doc in enumerate(retrieved_documents, 1):
        summary = (
            "Summary: " + doc[:150] + "..." if len(doc) > 150 else "Summary: " + doc
        )
        summarized_documents["Document " + str(i)] = summary
    return summarized_documents


app = Flask(__name__)
IMG_FOLDER = os.path.join("static", "IMG")

app.config["UPLOAD_FOLDER"] = IMG_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    Flask_Logo = os.path.join(app.config["UPLOAD_FOLDER"], "logo.png")
    return render_template("index.html",logo=Flask_Logo)


@app.route("/search", methods=["POST"])
def search():
    if request.method == "POST":
        query = request.form.get("input_text")
        query_vector = count_vectorizer.transform([query])
        query_tfidf = tfidf_transformer.transform(query_vector)
        similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
        top_indices = similarity_scores.argsort()[0][::-1]
        top_n = 5
        retrieved_documents = [df["Subtitles"][idx] for idx in top_indices[:top_n]]
        summarized_docs = generate_summarized_documents(retrieved_documents)
        Flask_Logo = os.path.join(app.config["UPLOAD_FOLDER"], "logo.png")
        return render_template(
            "index.html",
            query=query,
            results=summarized_docs,
            logo=Flask_Logo,
        )


if __name__ == "__main__":
    app.run(debug=True)
