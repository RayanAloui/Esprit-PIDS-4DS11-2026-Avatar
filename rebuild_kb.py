#!/usr/bin/env python3
"""
Rebuild ALIA knowledge base with improved indexing
This will delete the old database and create a new one with better document structure
"""

import os
import shutil
import pandas as pd
import asyncio

def rebuild_knowledge_base():
    print("\n" + "="*60)
    print("  REBUILD ALIA KNOWLEDGE BASE")
    print("="*60 + "\n")
    
    # Check if CSV exists
    csv_path = "vital_products.csv"
    if not os.path.exists(csv_path):
        print(f"✗ {csv_path} not found!")
        print("  Make sure the CSV file is in the current directory")
        return False
    
    # Import the improved RAG module
    try:
        import RAG_gemini as rag_mod
        print("✓ RAG module loaded\n")
    except ImportError as e:
        print(f"✗ Failed to import RAG module: {e}")
        return False
    
    # Load CSV
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path).fillna("")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    print(f"✓ Loaded {len(df)} products\n")
    
    # Show sample of what we have
    print("Sample products in database:")
    for i, row in df.head(5).iterrows():
        name = row.get('name', 'Unknown')
        indications = str(row.get('indications', ''))[:50]
        print(f"  - {name}")
        print(f"    Indications: {indications}...")
    print()
    
    # Delete old database
    db_path = "./alia_knowledge_db"
    if os.path.exists(db_path):
        response = input(f"Delete existing database at {db_path}? (yes/no): ")
        if response.lower() == 'yes':
            print(f"Deleting {db_path}...")
            shutil.rmtree(db_path)
            print("✓ Old database deleted\n")
        else:
            print("Keeping existing database. Rebuild cancelled.")
            return False
    
    # Build new documents with improved structure
    print("Building documents with improved structure...")
    docs = rag_mod.DataProcessor.build_documents(df)
    print(f"✓ Built {len(docs)} documents\n")
    
    # Show sample document
    print("Sample document structure:")
    print("-" * 60)
    print(docs[0].page_content[:300] + "...")
    print("-" * 60)
    print()
    
    # Create new knowledge base
    print("Creating new knowledge base...")
    print("(This will take a few minutes - Ollama needs to embed all documents)\n")
    
    manager = rag_mod.KnowledgeManager()
    manager.load_or_create(docs)
    
    print("\n✓ Knowledge base rebuilt successfully!\n")
    
    # Test retrieval
    print("Testing retrieval with sample queries...")
    print("="*60)
    
    test_queries = [
        "anémie",
        "fatigue",
        "rhume",
        "digestion"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = manager.retriever.invoke(query)
        print(f"Retrieved {len(results)} documents")
        if results:
            top_product = results[0].metadata.get('name', 'Unknown')
            print(f"Top result: {top_product}")
    
    print("\n" + "="*60)
    print("  REBUILD COMPLETE!")
    print("="*60)
    print("\n✓ You can now use the improved RAG module")
    print("  Update alia_server.py to import RAG_gemini_improved\n")
    
    return True

if __name__ == "__main__":
    success = rebuild_knowledge_base()
    exit(0 if success else 1)
