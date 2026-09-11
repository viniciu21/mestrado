import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DOCS_DIR = "./src/docs"
DB_DIR = "./src/chroma_db"
OLLAMA_URL = "http://localhost:11434"

BATCH_SIZE = 50


def build_knowledge_base():

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(
            f"Pasta '{DOCS_DIR}' criada. "
            "Adicione seus PDFs nela e execute novamente."
        )
        return

    print("1. Carregando arquivos PDF...")

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    if not documents:
        print(f"Nenhum PDF encontrado na pasta '{DOCS_DIR}'.")
        return

    print(f"-> {len(documents)} página(s) carregada(s).")

    print("2. Dividindo o texto em fragmentos (chunks)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"-> {len(chunks)} fragmentos criados.")

    print("3. Conectando ao Ollama...")

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding",
        base_url=OLLAMA_URL,
    )

    print("4. Testando embedding...")

    test_embedding = embeddings.embed_query("Teste de embedding")

    print(f"-> Embedding gerado: {len(test_embedding)} dimensões.")

    print("5. Criando ChromaDB...")

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    print("6. Inserindo documentos...")

    for i in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[i:i + BATCH_SIZE]

        print(
            f"-> Processando {i + 1}-{min(i + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)}"
        )

        vectorstore.add_documents(batch)

    print(
        f"Base de conhecimento criada com sucesso "
        f"na pasta '{DB_DIR}'!"
    )


if __name__ == "__main__":
    build_knowledge_base()