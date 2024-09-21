# Assuming you're using a chroma db client for semantic search
from chromadb import Client
import openai
import pandas as pd
client = openai.OpenAI()

embeddings_model = "text-embedding-3-large"


def get_embeddings(inputs: list[str], model_name: str = embeddings_model):
    response = client.embeddings.create(
        input=inputs,
        model=model_name
    )
    return [embedding.embedding for embedding in response.data]


def load_df(file_path: str = '../raw/df_with_embeddings.pkl'):
    return pd.read_pickle(file_path)


class MovieData:
    def __init__(self, movies_df: pd.DataFrame):
        self.df = movies_df
        batch_size = 41666

        print("Loading vector db")
        self.db_client = Client()  # Assuming chromadb client initialization

        # If collection exists, use it
        try:
            self.collection = self.db_client.get_collection(name="movies")
            entries = self.collection.count()
            if entries != len(self.df):
                print("Collection does not match the number of movies")
                raise ValueError(
                    "Collection does not match the number of movies")
        except:
            self.collection = self.db_client.get_or_create_collection(
                name="movies")

            ids = self.df.index.tolist()
            id_strings = [str(id) for id in ids]
            embeddings = self.df['embedding'].tolist()

            for i in range(0, len(ids), batch_size):
                print(f"Processing batch {i} of {len(ids)}")
                batch_ids = id_strings[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                self.collection.upsert(
                    ids=batch_ids, embeddings=batch_embeddings)

    def semantic_search(self, query: str, k: int = 10):
        response = client.embeddings.create(
            input=query,
            model=embeddings_model
        )
        query_embedding = response.data[0].embedding
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=k)
        ids: list[str] = results['ids'][0]
        # df from the ids
        ids = [int(i) for i in ids]

        results_df = self.df.loc[ids]
        results_df["distance"] = results['distances'][0]
        return results_df
