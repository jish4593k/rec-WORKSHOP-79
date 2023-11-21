import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_html_content(url, tag_name, tag_attrs):
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    return soup.find_all(name=tag_name, attrs=tag_attrs)

def preprocess_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    encoded_text = tokenizer.encode(text, add_special_tokens=True, return_tensors='tf')
    return encoded_text

def calculate_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def plot_similarity_scores(scores):
    plt.bar(range(len(scores)), scores, align='center')
    plt.xticks(range(len(scores)), [f'Document {i}' for i in range(len(scores))])
    plt.xlabel('Documents')
    plt.ylabel('Cosine Similarity Score')
    plt.title('Cosine Similarity Scores Between Documents')
    plt.show()

def main():
    documents = [
        {'url': "https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=105&oid=008&aid=0004159617",
         'name': "div",
         'attr': {"id": "articleBodyContents"}},
        # Add more documents as needed
    ]

    document_texts = [get_html_content(doc['url'], doc['name'], doc['attr'])[0].get_text() for doc in documents]

    # Preprocess and get embeddings using BERT
    embeddings = [TFBertModel.from_pretrained('bert-base-multilingual-cased')(preprocess_text(text))[0][:, 0, :].numpy()
                  for text in document_texts]

    # Calculate cosine similarity between documents
    similarity_scores = [calculate_similarity(embeddings[0], embedding) for embedding in embeddings]

    # Print similarity scores
    for i, score in enumerate(similarity_scores):
        print(f"Similarity between Document 0 and Document {i}: {score:.4f}")

    # Plot the similarity scores
    plot_similarity_scores(similarity_scores)

if __name__ == "__main__":
    main()
