from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


class DataIngestion():

    def __init__(self):
        pass

    def initiate_data_ingestion(self,data):

        # text loader
        try:
            loader = TextLoader(data)

            documents = loader.load()
            
        except FileNotFoundError:
            print("File are not present",data)
            return []

        splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 100,
                    chunk_overlap = 20
                )

        # splitting documents into multiple chunks
        chunks = splitter.split_documents(documents)

        print("length of documnets-",len(documents))
        print()
        print("length of chunks-",len(chunks))

        return chunks

if __name__=="__main__":
    crick = DataIngestion()
    data = "cricket_rag//data//cricket_basics.txt"
    chunks = crick.initiate_data_ingestion(data)

   


        
    