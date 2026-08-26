import time
import vertexai
from vertexai.preview import rag
from vertexai.preview.rag.utils import resources as rr

PROJECT_ID = "qwiklabs-gcp-04-17cb16fe3675"
LOCATION = "us-central1"
GCS_PATH = "gs://bic-mac-storage-17cb16fe3675/rag/pg49513.txt"

PARSING_PROMPT = (
    "Extract useful knowledge, domain facts, and medical background from this document. "
    "Omit boilerplate and header information. Output clean, self-contained prose."
)


def create_rag_corpus():
    print(f"Initializing Vertex AI for RAG: project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # 1. Switch region RAG managed DB to serverless mode
    print("Setting RAG Engine config to serverless mode...")
    config_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragEngineConfig"
    try:
        rag.update_rag_engine_config(
            rag_engine_config=rag.RagEngineConfig(
                name=config_name,
                rag_managed_db_config=rag.RagManagedDbConfig(mode=rr.Serverless()),
            )
        )
        print("Serverless RAG DB mode updated.")
    except Exception as e:
        print(f"Notice during update_rag_engine_config: {e}")

    # 2. Create the corpus
    print("Creating RAG corpus...")
    corpus = rag.create_corpus(
        display_name="bic-mac-knowledge-corpus",
        embedding_model_config=rag.EmbeddingModelConfig(
            publisher_model="publishers/google/models/text-embedding-005"
        ),
    )
    print(f"✅ RAG Corpus Created: {corpus.name}")

    # 3. Import, chunk, and embed the file
    print(f"Importing and indexing {GCS_PATH} into corpus...")
    resp = rag.import_files(
        corpus_name=corpus.name,
        paths=[GCS_PATH],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
        ),
        llm_parser=rag.LlmParserConfig(
            model_name="gemini-2.5-flash",
            custom_parsing_prompt=PARSING_PROMPT,
        ),
    )
    print(f"✅ Import complete! Imported files count: {getattr(resp, 'imported_rag_files_count', 'N/A')}")
    print(f"CORPUS_NAME: {corpus.name}")
    return corpus.name


if __name__ == "__main__":
    create_rag_corpus()
