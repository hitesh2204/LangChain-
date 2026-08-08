from langchain_community.document_loaders import DirectoryLoader,TextLoader

loader = DirectoryLoader(
    path = "RAG/document_loader/directory",
    glob = '*.txt',
    loader_cls = TextLoader,
    show_progress=True
)

documents = loader.load()
print(len(documents))
print()

for document in documents:
    print(document.page_content)
    print(document.metadata)