import re
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

class RAGAssistant:
    def __init__(self, csv_path="vital_products.csv", persist_dir="./chroma_vital_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self._init_vectorstore()
        self._init_chain()
        print("RAG pret!")
    
    def _init_vectorstore(self):
        if os.path.exists(self.persist_dir):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings
            )
            print(f"Base chargee depuis '{self.persist_dir}'")
        else:
            self._build_vectorstore()
    
    def _build_vectorstore(self):
        print("Creation de la base...")
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
        df = df.fillna("Non spécifié")
        
        documents = []
        for _, row in df.iterrows():
            content = (
                f"Nom du produit: {row['name']}\n"
                f"Categories: {row['categories']}\n"
                f"Indications: {row['indications']}\n"
                f"Forme: {row['forme']}\n"
                f"Infos produit: {row['infos_produit']}\n"
                f"Classe: {row['classe']}\n"
                f"URL: {row['url']}"
            )
            metadata = {
                "name": row["name"],
                "categories": row["categories"],
                "forme": row["forme"],
                "classe": row["classe"],
                "url": row["url"],
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"Base sauvegardee dans '{self.persist_dir}'")
    
    def _init_chain(self):
        self.llm = OllamaLLM(model="deepseek-r1", temperature=0.1)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        
        self.expand_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Tu es un expert en produits de sante.
Reecris la question en 4 variantes en francais.
Retourne UNIQUEMENT les 4 variantes, une par ligne.

Question: {question}

Variantes:""",
        )
        
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Tu es un assistant catalogue pour Vital Tunisie.

Contexte:
{context}

Question: {question}

Produits correspondants:""",
        )
    
    def _strip_think(self, text):
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
   
    def _retrieve_docs(self, question):
        docs = self.retriever.invoke(question)
        return docs
    
    def ask(self, question):
        print(f"\nQuestion: {question}")
        docs = self._retrieve_docs(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        reponse = self._strip_think(
            (self.rag_prompt | self.llm | StrOutputParser()).invoke({
                "context": context,
                "question": question,
            })
        )
        return reponse

# Pour test indépendant
if __name__ == "__main__":
    rag = RAGAssistant()
    reponse = rag.ask("Quels produits pour les douleurs?")
    print(f"Reponse: {reponse}")