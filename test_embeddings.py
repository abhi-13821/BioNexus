from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Breast cancer is one of the most common cancers.",
    "EGFR mutations are associated with lung cancer.",
    "Aspirin may reduce inflammation."
]

embeddings = model.encode(texts)

print("Model loaded successfully!")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")

print("\nFirst 10 values of the first embedding:")
print(embeddings[0][:10])