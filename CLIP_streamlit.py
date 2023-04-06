import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd 
import matplotlib.pyplot as plt
import os  
#import cv2
import torch
import requests
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity


# Define list of available models
applications = ["Image Search By Text", "Image Search By Image"]

# Define dataset of sentences for MPNet
file_path_clip = "data/df_clip_embeddings.csv"
df_clip= pd.read_csv(file_path_clip)
#df_mpnet['embeddings '] = df_mpnet['embeddings '].apply(ast.literal_eval)

# Define dataset of sentences for MPNet
#DOCUMENTS_EMBEDDINGS_PATH = "data"  # a folder with all the documents embeddings. within this folder, one csv file include multiple documents embedding of the same run
#COLUMN_EMBEDDINGS = "embedding"  # the embedding column name in the documents embedding file.
model = SentenceTransformer('clip-ViT-B-32')

# Set up Streamlit app
st.title("Image Search Engine")

def cosine_similarity_clip(a,b):
    dot_product = np.dot(a, b)
    magnitude_a = np.sqrt(np.dot(a, a))
    magnitude_b = np.sqrt(np.dot(b, b))
    cos_sim = dot_product / (magnitude_a * magnitude_b)
    return cos_sim

application_choice = st.selectbox("Choose an option:", applications)
# Calculate embeddings and get similar sentences
#if st.button("Enter"):
if application_choice=="Image Search By Text":
    # Replace with code to load selected model
    desired_input = st.text_input("Enter a text query:")
    #if st.button("Find Related Photos:"):


elif application_choice=="Image Search By Image":
    st.write("Please select an image file")
    desired_input=None
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_bytes = uploaded.read()
        image = BytesIO(image_bytes)
        desired_input = Image.open(image)
        #st.write("type",desired_input)
    


if desired_input:
    input_vector = np.array(model.encode(desired_input))
    similarity_scores =  [cosine_similarity_clip(input_vector,np.fromstring(embedding[1:-1], sep=' ')) for embedding in df_clip['img_embeddings'].tolist()]
    top_3_indices = np.argsort(-np.array(similarity_scores))[:3]
    selected_rows = df_clip.loc[top_3_indices.tolist(), 'photo_image_url']
    for i,url in enumerate(selected_rows):
        st.write("Resulted Image Number",i+1,'is:\n')
        response = requests.get(url,stream=True)
        img = Image.open(BytesIO(response.content))
        st.image(img)
      
        
        

 