from cricket_rag.src.vector_store import VectoreStore

class Retriever():

    def __init__(self):
        pass

    def initiate_retriever(self,vector_obj):

        #creating the retriever.
        retriever_obj = vector_obj.as_retriever(search_type ='mmr',search_kwargs={'k':2})

        return retriever_obj


if __name__=="__main__":
    # Create vector store object
    vector_store = VectoreStore()

    # Create retriever object
    retriever_class = Retriever()

    # Load existing FAISS vector store
    vector_db = vector_store.load_vector_store()

    # Create retriever
    retriever = retriever_class.initiate_retriever(vector_db)

    print("Retriever created successfully!")

    # Test retriever
    query = "How many players are there in a cricket team?"

    results = retriever.invoke(query)

    print("\nRetrieved Documents:\n")

    for result in results:
        print(result.page_content)
        print("-" * 50)
