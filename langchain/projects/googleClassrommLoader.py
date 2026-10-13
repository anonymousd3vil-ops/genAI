from langchain_google_classroom import GoogleClassroomLoader

loader = GoogleClassroomLoader()
# Opens browser for OAuth consent on first run
# docs = loader.load()

# for doc in docs:
#     print(doc.page_content)

def get_classroom_documents():
    """Fetch documents from the existing Google Classroom loader."""
    return loader.load()


if __name__ == "__main__":
    docs = get_classroom_documents()
    print(f"Loaded {len(docs)} classroom documents.")

    for doc in docs:
        print(doc.metadata.get("title", "Untitled"))

# print(docs)
# classroom_data = []

# for doc in docs:
#     classroom_data.append({
#         "course_id": doc.metadata.get("course_id"),
#         "course_name": doc.metadata.get("course_name"),
#         "content_type": doc.metadata.get("content_type"),
#         "title": doc.metadata.get("title"),
#         "item_id": doc.metadata.get("item_id"),
#         "created_time": doc.metadata.get("created_time"),
#         "updated_time": doc.metadata.get("updated_time"),
#         "due_date": doc.metadata.get("due_date"),
#         "max_points": doc.metadata.get("max_points"),
#         "url": doc.metadata.get("alternate_link"),
#         "state": doc.metadata.get("state"),
#         "content": doc.page_content
#     })

# import json

# with open("classroom_data.json", "w", encoding="utf-8") as file:
#     json.dump(
#         classroom_data,
#         file,
#         indent=4,
#         ensure_ascii=False
#     )

# print("Classroom data saved successfully.")
# print(classroom_data)
# for i, doc in enumerate(classroom_data, start=1):
#     print(f"\n{'=' * 60}")
#     print(f"Document {i}")
#     print(f"{'=' * 60}")

#     print("\nMetadata:")
#     print(doc.metadata)

#     print("\nContent:")
#     print(doc.page_content)