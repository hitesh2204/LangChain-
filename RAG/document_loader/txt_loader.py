from langchain_community.document_loaders import TextLoader

cricket_loader = TextLoader("RAG/document_loader//cricket.txt")
football_loader = TextLoader("RAG/document_loader//football.txt")


cricket_documents = cricket_loader.load()
football_documnets = football_loader.load()

#print(type(documents))
#print(len(documents))
documents = cricket_documents + football_documnets
print(len(documents))
print()
print(documents)
print()
print("Football documents -",documents[1])
print("page_content-",documents[1].page_content)
print()
print("metadata-",documents[1].metadata)
print(type(documents[0]))
print("complete documents ",cricket_documents)
print("complete documents",football_documnets)
print()
print("contain inside documents ",cricket_documents[0])
print("content inside documents",football_documnets[0])
print()
print("page content inside documents",cricket_documents[0].page_content)
print("page content inside documenst",football_documnets[0].page_content)
print()
print("metadata inside documents",cricket_documents[0].metadata)
print("metadata inside documents",football_documnets[0].metadata)