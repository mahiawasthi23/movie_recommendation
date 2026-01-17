import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
    page_icon="🎬"
)

@st.cache_data
def load_data():
    data = pd.read_csv("data/movies.csv").head(5000)
    data["clean_title"] = data["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    data["tags"] = data["genres"].fillna("")
    return data

movies = load_data()


@st.cache_resource 
def create_similarity(data):
    cv = CountVectorizer(stop_words="english", max_features=1500)
    vectors = cv.fit_transform(data["tags"]).toarray().astype('float32') # Use float32 to save RAM
    return cosine_similarity(vectors)

similarity = create_similarity(movies)

def get_movie_index(title):
    try:
        return movies[movies["clean_title"] == title].index[0]
    except:
        return None

def recommend(movie_index, n=5):
    if movie_index is None:
        return []
    
    distances = similarity[movie_index]
    scores = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:n+1]
    return [movies.iloc[i[0]]["clean_title"] for i in scores]


st.markdown("""
<style>
.stApp {
    background-color: #0f0f0f;
    color: white;
}
section[data-testid="stAppViewContainer"] {
    padding-top: 40px;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #aaa;
    margin-bottom: 25px;
}
.stButton > button {
    background: transparent;
    border: 2px solid #e50914;
    color: #e50914;
    padding: 10px 35px;
    font-size: 16px;
    border-radius: 8px;
    transition: 0.3s;
}
.stButton > button:hover {
    background: #e50914;
    color: white;
}
.movie-card {
    background: #1c1c1c;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    transition: 0.3s;
    min-height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.movie-card:hover {
    background: #e50914;
    transform: scale(1.08);
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>🎬 Movie Recommender System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Select a movie and get recommendations</div>", unsafe_allow_html=True)

center = st.columns([2, 6, 2])
with center[1]:
    selected_movie = st.selectbox(
        "Choose a movie",
        [""] + list(movies["clean_title"].unique()),
        label_visibility="collapsed"
    )

btn_col = st.columns([2, 4, 4])
with btn_col[1]:
    btn = st.button("Recommend")

if btn:
    if selected_movie == "":
        st.warning("Please select a movie first!")
    else:
        idx = get_movie_index(selected_movie)
        recs = recommend(idx)

        if not recs:
            st.error("Sorry, could not find any recommendations for this movie.")
        else:
            st.markdown("### 🍿 Recommended Movies")
            cols = st.columns(5)
            for col, title in zip(cols, recs):
                with col:
                    st.markdown(
                        f"<div class='movie-card'>{title}</div>",
                        unsafe_allow_html=True
                    )