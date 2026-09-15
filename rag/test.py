from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

points, next_page = client.scroll(
    collection_name="mem0_vivek_entities",
    limit=100,
    with_payload=True,
    with_vectors=False
)

for point in points:
    print("\nID:", point.id)
    print("PAYLOAD:", point.payload)