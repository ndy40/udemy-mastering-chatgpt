import dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

dotenv.load_dotenv()

loader = TextLoader('facts.txt')
embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator='\n', chunk_size=200, chunk_overlap=50
)
docs = loader.load_and_split(text_splitter=text_splitter)
db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory='emb ')

results = db.similarity_search_with_score('what is an interesting fact about the English language?')

for result in results:
    print('\n', result[1], '\n', result[0].page_content)