import requests
import tempfile
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFium2Loader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pypdfium2 as pdfium
from constants import chunk_size, chunk_overlap, number_snippets_to_retrieve

def download_and_index_pdf(urls: list[str]) -> FAISS:
    """
    Download and index a list of PDFs based on the URLs.
    """

    def __update_metadata(pages, url):
        """
        Add title and original URL to the document metadata.
        """
        for page in pages:
            try:
                local_path = page.metadata['source']
                pdf = pdfium.PdfDocument(local_path)
                title = pdf.get_metadata_dict().get('Title', url)
                page.metadata['source'] = url
                page.metadata['title'] = title
            except Exception as e:
                print(f"Metadata update failed for {url}: {e}")
        return pages

    all_pages = []
    for url in urls:
        try:
            # Download PDF to a temporary file
            response = requests.get(url)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_pdf_path = tmp_file.name

            # Load and split PDF
            loader = PyPDFium2Loader(tmp_pdf_path)
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            pages = loader.load_and_split(splitter)

            # Update metadata
            pages = __update_metadata(pages, url)
            all_pages.extend(pages)

        except Exception as e:
            print(f"Failed to process {url}: {e}")

    # Create FAISS index
    faiss_index = FAISS.from_documents(all_pages, OpenAIEmbeddings())
    return faiss_index


def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    """
    Search a FAISS index, using the passed query.
    """
    return faiss_index.similarity_search(query, k=top_k)
