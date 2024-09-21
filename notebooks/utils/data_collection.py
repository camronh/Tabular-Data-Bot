# Assuming you're using a chroma db client for semantic search
from chromadb import Client
import openai
import pandas as pd
from langchain_core.tools import BaseTool
import numpy as np
from pydantic.v1 import BaseModel, Field
from typing import Type, Optional, Literal


client = openai.OpenAI()

chroma_client = Client()

embeddings_model = "text-embedding-3-large"

ColumnOptions = Literal['title', 'vote_average', 'vote_count', 'status', 'release_date',
                        'revenue', 'runtime', 'budget', 'imdb_id', 'original_language',
                        'original_title', 'overview', 'popularity', 'tagline', 'genres',
                        'production_companies', 'production_countries', 'spoken_languages',
                        'cast', 'director', 'director_of_photography', 'writers', 'producers',
                        'music_composer', 'imdb_rating', 'imdb_votes', 'embedding',
                        'embedding_norm']

SortOptions = Literal['rating']


class Sort(BaseModel):
    sort: SortOptions
    ascending: bool = True


class MovieSearch(BaseModel):
    query: Optional[str] = Field(
        description="A semantic search for movie data")
    sort: Optional[Sort] = Field(
        description="Sort the results by a column")


def get_embeddings(inputs: list[str], model_name: str = embeddings_model):
    response = client.embeddings.create(
        input=inputs,
        model=model_name
    )
    return [embedding.embedding for embedding in response.data]


def load_df(file_path: str = '../raw/top_50000.pkl') -> pd.DataFrame:
    print("Loading df")
    return pd.read_pickle(file_path)


def df_to_llm(df: pd.DataFrame) -> list[dict]:
    """Parse out only what is relevant to the LLM"""
    movies = []
    for index, row in df.iterrows():
        movies.append({
            "title": row["title"],
            "status": row["status"],
            "release_date": row["release_date"],
            "revenue": row["revenue"],
            "runtime": row["runtime"],
            "budget": row["budget"],
            "url": f"https://www.imdb.com/title/{row['imdb_id']}",
            "original_language": row["original_language"],
            "original_title": row["original_title"],
            "overview": row["overview"],
            "popularity": row["popularity"],
            "tagline": row["tagline"],
            "genres": row["genres"],
            "production_companies": row["production_companies"],
            "production_countries": row["production_countries"],
            "spoken_languages": row["spoken_languages"],
            "cast": row["cast"],
            "director": row["director"],
            "writers": row["writers"],
            "producers": row["producers"],
            "rating": row["imdb_rating"],
            "votes": row["imdb_votes"],
        })
    return movies


class MovieSearchTool(BaseTool):
    name: str = "movie_data_search"
    description: str = "Use this tool to search for movies using natural language"
    args_schema: Type[BaseModel] = MovieSearch

    # Define it as part of the class attribute, not the Pydantic model fields
    df: pd.DataFrame = None

    def __init__(self):
        super().__init__()  # Ensure you call the parent class' initializer if needed
        self.df = load_df("../raw/top_50000.pkl")  # Load the dataframe here

    def semantic_search(self, query: str, k: int = 10):
        # Assuming you already have a model to generate embeddings for the query
        response = client.embeddings.create(
            input=query,
            model=embeddings_model
        )
        print("Got query embeddings")
        query_embedding = np.array(response.data[0].embedding)

        # Precompute the norm of the query embedding
        query_norm = np.linalg.norm(query_embedding)

        # Compute dot products and cosine similarities
        self.df['dot_product'] = self.df['embedding'].apply(
            lambda x: np.dot(query_embedding, x))
        self.df['similarity'] = self.df['dot_product'] / \
            (self.df['embedding_norm'] * query_norm)

        # Sort by similarity and get top k results
        results_df = self.df.sort_values(by='similarity', ascending=False)
        results_df = results_df.head(k)

        # Sort by vote count
        results_df = results_df.sort_values(by='vote_count', ascending=False)

        # Return top k results
        return results_df

    def _run(self, query: str):
        return df_to_llm(self.semantic_search(query))
