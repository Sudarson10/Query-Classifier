import streamlit as st
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
with open(r'C:\Users\sudar\OneDrive\Desktop\miniprojects\Query Classifier\sampled_ecommerceDataset.pkl', 'rb') as f:
    data = pickle.load(f)

similarity=data['vectors']

sampled_df=pd.read_csv(r"C:\Users\sudar\OneDrive\Desktop\miniprojects\Query Classifier\sampled_ecommerceDataset.csv")

# Predefined category descriptions
categories_info = {
    "Household": """**Household:**
    
    - Furniture (e.g., sofas, tables, chairs)
    - Kitchenware (e.g., utensils, cookware, appliances)
    - Home decor (e.g., rugs, curtains, wall art)
    - Cleaning supplies (e.g., detergents, mops, vacuum cleaners)
    - Bedding (e.g., sheets, blankets, pillows)
    - Grocery items (e.g., vegetables, fruits, etc.)""",

    "Books": """**Books:**
    
    - Fiction (e.g., novels, short stories)
    - Non-fiction (e.g., biographies, self-help, history)
    - Academic textbooks
    - Children's books
    - E-books and audiobooks""",

    "Clothing & Accessories":"""**Clothing & Accessories:**
    
    - Apparel (e.g., shirts, pants, dresses, coats)
    - Footwear (e.g., shoes, boots, sandals)
    - Accessories (e.g., bags, belts, hats, scarves)
    - Jewelry (e.g., necklaces, bracelets, rings)""",

    "Electronics": """**Electronics:**
    
    - Consumer electronics (e.g., smartphones, tablets, laptops)
    - Home entertainment (e.g., TVs, gaming consoles, speakers)
    - Wearable technology (e.g., smartwatches, fitness trackers)
    - Appliances (e.g., refrigerators, microwaves, washing machines)
    - Computer components and accessories (e.g., monitors, keyboards, external drives)"""
}
dim=similarity.shape[1]

index=faiss.IndexFlatL2(dim)

index.add(similarity)

encoder=SentenceTransformer("all-mpnet-base-v2")


# Function to classify the query
def classify_query(svec):
    a, b = index.search(svec, k=1)
    ans = sampled_df.loc[b[0]]
    return ans.category.values[0]


st.title("Context-Sensitive Query Classification System")
st.header("Description")
st.subheader("The Context-Sensitive Query Classification System is an intelligent application designed to categorize user queries based on the context of the question.")
# Displaying category information
st.sidebar.markdown(categories_info["Household"])
st.sidebar.markdown(categories_info["Books"])
st.sidebar.markdown(categories_info["Clothing & Accessories"])
st.sidebar.markdown(categories_info["Electronics"])

# Input box for user query
user_query = st.text_input("Enter your query:")

# Classify the query and display the result
if user_query:
    svec = encoder.encode(user_query)
    svec = np.array(svec).reshape(1, -1)
    category = classify_query(svec)
    st.write(f"Category: {category}")




