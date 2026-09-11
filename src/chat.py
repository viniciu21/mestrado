from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "./src/chroma_db"
OLLAMA_URL = "http://localhost:11434"

def format_docs(docs):
    # print("\n" + "="*40)
    # print("🔍 TRECHOS RECUPERADOS DO BANCO:")
    # for i, doc in enumerate(docs):
    #     # O metadata pega o nome do PDF e a página (se disponível)
    #     origem = doc.metadata.get('source', 'Desconhecida')
    #     pagina = doc.metadata.get('page', '?')
    #     print(f"\n--- TRECHO {i+1} | {origem} (Pág {pagina}) ---")
    #     print(doc.page_content)
    # print("="*40 + "\n")
    return "\n\n".join(doc.page_content for doc in docs)

def start_chatbot():
    print("Conectando aos modelos no Ollama e carregando a base vetorial...")
    
    # Embeddings idênticos aos usados na criação da base
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding",
        base_url=OLLAMA_URL,
    )
    
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("O que voltou", retriever)
    # LLM Principal
    llm = ChatOllama(
        model="qwen2.5:14b",
        base_url=OLLAMA_URL,
        temperature=0.6,
        num_ctx=16384 # 16k de contexto para aproveitar os 16GB da RTX 5070 Ti
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente especializado. Responda utilizando apenas "
                   "o contexto abaixo. Se a resposta não estiver no contexto, diga "
                   "que não possui essa informação.\n\n"
                   "Contexto:\n{context}"),
        ("human", "{question}"),
    ])

    # Pipeline LCEL limpo e direto
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- Chatbot RAG (LCEL) com Qwen 2.5 14B Ativo! (Digite 'sair' para encerrar) ---\n")
    
    while True:
        query = input("Você: ").strip()
        if not query:
            continue
        if query.lower() in ["sair", "exit", "quit"]:
            break

        # Invocação direta da string
        response = rag_chain.invoke(query)
        
        print(f"\nBot: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    start_chatbot()