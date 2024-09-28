# Tabular Data Bot

This is an experiment on doing RAG with tabular data. We focus on "Eval Driven Development" where we start with an evaluation dataset and focus on improving the scores of these evals.

## Process

Here is the process we followed to iterate on the bot.

1. [Data Collection](./notebooks/1_Data_Collection.ipynb) - We started by cleaning the data. We removed any bad data and normalized the movie data. Then we added a very simple semantic search functionality to it by adding an embeddings column and an `embedding_norm` column that speeds up the semantic search by a lot.

2. [Build the Agent](./notebooks/2_Agent.ipynb) - We build a super basic agent that can invoke a single tool for searching movies. We use [LangGraph](https://www.langchain.com/langgraph) to orchestrate the agent in the notebook before moving it to the [`langgraph.py`](./notebooks/utils/langgraph.py) file.

3. [Build the Ground Truth Dataset](./notebooks/3_Dataset.ipynb) - We come up with some example questions that we expect users to ask. We then run the agent against the dataset with [LangSmith](https://www.langchain.com/langsmith) to store the traces. Then we add those traces to a [LangSmith Annotation queue](https://docs.smith.langchain.com/how_to_guides/human_feedback/annotation_queues), which makes it easy for use to manually correct the answers before adding them to a dataset. I went through and corrected the answers for all of the examples by hand before adding them to a [LangSmith Dataset](https://smith.langchain.com/public/c7fff8c1-b060-4746-8a81-3ed9c2409a8a/d).

4. [Evaluation](./notebooks/4_Evals.ipynb) - We build an [LLM as a Judge](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) that basically, for each example, takes in the correct answer from our [ground truth dataset](https://smith.langchain.com/public/c7fff8c1-b060-4746-8a81-3ed9c2409a8a/d) and the answer that the agent outputs and decides if the answer is correct. This will allow us to automatically score the agent's performance because we have already defined what correct answers are.

5. Iterate - Now we have a baseline of how the agent performs. We have everything we need to start iterating. At this point we, run the evals and check the results for which ones failed. We decide based on value which features we want to add and in which order. We implement those features and run evals again to see how the results have improved. 

6. Expand the Dataset - Once we get the score to a level that we are happy with, our job should be to decrease the score by expanding the dataset. We want to add new examples that the agent can't solve correctly and then repeat the process starting from step 3.
