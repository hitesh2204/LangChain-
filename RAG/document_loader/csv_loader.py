from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("RAG/document_loader/player.csv")
documents = loader.load()

print(len(documents))

for i in documents:
    print()
    print("page_content-",i.page_content)
    print("metadata-",i.metadata)