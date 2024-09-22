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

SortOptions = Literal['rating', "votes", "release_date", "revenue"]

GenreOptions = Literal['Action', 'Science Fiction',
                       'Adventure',
                       'Drama',
                       'Crime',
                       'Thriller',
                       'Fantasy',
                       'Comedy',
                       'Romance',
                       'Western',
                       'Mystery',
                       'War',
                       'Animation',
                       'Family',
                       'Music',
                       'Horror',
                       'History',
                       'TV Movie',
                       'Documentary']

LanguageOptions = Literal['en', 'ko', 'fr', 'ja', 'it', 'es', 'pl', 'pt', 'hi', 'tr', 'da',
                          'de', 'cn', 'id', 'zh', 'sv', 'el', 'lt', 'ru', 'fi', 'sr', 'fa',
                          'ar', 'no', 'th', 'te', 'nl', 'la', 'hu', 'he', 'is', 'uk', 'ro',
                          'gl', 'eu', 'et', 'bs', 'bn', 'ga', 'sh', 'cs', 'hy', 'km', 'tn',
                          'ml', 'mk', 'ka', 'kn', 'ku', 'ta', 'ca', 'vi', 'tl', 'dz', 'sk',
                          'xx', 'kk', 'sw', 'lv', 'tt', 'bo', 'ur', 'mi', 'ps', 'mn', 'wo',
                          'sl', 'sc', 'af', 'ms', 'se', 'cy', 'iu', 'hr', 'mr', 'bm', 'bg',
                          'lo', 'yo', 'am', 'xh', 'yi', 'sq', 'qu', 'eo', 'gu', 'pa', 'os',
                          'ln', 'zu', 'ig', 'nb', 'st', 'ne', 'mt', 'as', 'ak', 'rw']


class Sort(BaseModel):
    column: SortOptions = Field(
        description="The column to sort by")
    ascending: bool = Field(
        description="Whether to sort in ascending order")


class MovieSearch(BaseModel):
    query: Optional[str] = Field(
        description="A semantic search for movie data")
    sort: Optional[Sort] = Field(
        description="Sort the results by a column")
    cast: Optional[list[str]] = Field(
        description="A list of actors or actresses that the movie must include")
    director: Optional[str] = Field(
        description="The director of the movie")
    genres: Optional[list[GenreOptions]] = Field(
        description="A list of genres that the movie must include.")
    original_language: Optional[LanguageOptions] = Field(
        description="The original language of the movie")


def get_embeddings(inputs: list[str], model_name: str = embeddings_model):
    response = client.embeddings.create(
        input=inputs,
        model=model_name
    )
    return [embedding.embedding for embedding in response.data]


def load_df(file_path: str = '../raw/top_50000.pkl') -> pd.DataFrame:
    print("Loading df")
    return pd.read_pickle(file_path)


def filter_df(df: pd.DataFrame, column: str, contents: list[str]):
    """Filter the dataframe to only include rows that contain any of the contents"""
    # Get a df where the column is not null
    has_column = df[df[column].notna()]
    empty_df = pd.DataFrame()

    # Get all rows where the column contains any of the contents and combine them to a single df
    for content in contents:
        has_content = has_column[has_column[column].str.contains(content)]
        empty_df = pd.concat([empty_df, has_content])

    return empty_df


def sort_df(df: pd.DataFrame, sort: Sort) -> pd.DataFrame:
    if sort.column == "rating":
        return df.sort_values(by="imdb_rating", ascending=sort.ascending)

    elif sort.column == "votes":
        return df.sort_values(by="imdb_votes", ascending=sort.ascending)

    else:
        return df.sort_values(by=sort.column, ascending=sort.ascending)


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


NUMBER_OF_RESULTS = 10


class MovieSearchTool(BaseTool):
    name: str = "movie_data_search"
    description: str = "Use this tool to search for movies using natural language"
    args_schema: Type[BaseModel] = MovieSearch

    # Define it as part of the class attribute, not the Pydantic model fields
    df: pd.DataFrame = None

    def __init__(self):
        super().__init__()  # Ensure you call the parent class' initializer if needed
        self.df = load_df("../raw/top_50000.pkl")  # Load the dataframe here

    def semantic_search(self, query: str, k: int = 10, df: pd.DataFrame = None):
        if df is None:
            df = self.df

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
        df['dot_product'] = df['embedding'].apply(
            lambda x: np.dot(query_embedding, x))
        df['similarity'] = df['dot_product'] / \
            (self.df['embedding_norm'] * query_norm)

        # Sort by similarity and get top k results
        results_df = df.sort_values(by='similarity', ascending=False)
        results_df = results_df.head(k)

        # Sort by vote count
        results_df = results_df.sort_values(by='vote_count', ascending=False)

        # Return top k results
        return results_df

    def search_movies(self, query: str = None, sort: Sort = None,
                      cast: list[str] = None,
                      director: str = None,
                      genres: list[str] = None,
                      original_language: LanguageOptions = None):

        df = self.df.copy()

        # Filters
        if cast:
            df = filter_df(df, "cast", cast)

        if director:
            df = filter_df(df, "director", [director])

        if genres:
            df = filter_df(df, "genres", genres)

        if original_language:
            df = filter_df(df, "original_language", [original_language])


        # Sorting
        if query and sort:
            # Get top 20 and then sort
            df = self.semantic_search(query, 20, df)
            df = sort_df(df, sort)

        elif query:
            df = self.semantic_search(query, NUMBER_OF_RESULTS, df)

        elif sort:
            df = sort_df(df, sort).head(NUMBER_OF_RESULTS)

        return df

    def _run(self, query: str = None, sort: Sort = None, cast: list[str] = None, director: str = None,
             genres: list[str] = None, original_language: LanguageOptions = None):
        df = self.search_movies(
            query, sort, cast, director, genres, original_language)
        return df_to_llm(df)
