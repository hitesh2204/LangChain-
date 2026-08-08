from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("RAG/document_loader/Kundali_Report.pdf")

documents = loader.load()
# print(documents)
# print()
# print(len(documents))

for i in documents:
    print("page_content - ",i.page_content)
    print("metadat -",i.metadata)