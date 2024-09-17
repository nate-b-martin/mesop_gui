from langchain_chroma import Chroma
import os


def update_vector_store(self, store:Chroma):
    # Get the updated documents
    updated_docs = self.load_docs()
    
    # Update the embeddings of the new or updated documents
    for doc in updated_docs:
        embedding = embedding(doc)
        store.add_documents(doc, embedding)
    
    # Save the updated vector store
    store.save()

import os
import time


def find_recently_modified_files(minutes=5):
    current_time = time.time()
    five_minutes_ago = current_time - (minutes * 60)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_directory = os.path.join(project_dir, "test_data")

    recent_files = []

    for root, dirs, files in os.walk(test_directory):
        for file in files:
            file_path = os.path.join(root, file)
            modification_time = os.path.getmtime(file_path)
            
            if modification_time > five_minutes_ago:
                recent_files.append(file_path)

    return recent_files


if __name__ == "__main__":
    print(find_recently_modified_files())