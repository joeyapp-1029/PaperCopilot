import io
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter


class PDFWrapper:
    """
    A wrapper class that simplifies interactions with PDF files. It allows for extracting text,
    counting pages, and splitting the text into manageable chunks for further processing.
    """

    def __init__(self, reader):
        """
        Initializes the PDFWrapper instance with the PDF reader object, typically from PyPDF2.
        
        :param reader: a PDF reader object containing the pages of the PDF document.
        """
        self.pageNum = len(reader.pages)
        self.pages = {i: page.extract_text() for i, page in enumerate(reader.pages)}

    def get_num_pages(self):
        """
        Returns the total number of pages in the PDF document.
        
        :return: An integer representing the total number of pages.
        """
        return self.pageNum

    def get_page(self, page_num):
        """
        Retrieves the text of a specified page within the PDF document.
        
        :param page_num: The page number to retrieve.
        :return: Text content of the specified page or None if the page does not exist.
        """
        return self.pages.get(page_num)
    
    def get_all_text(self):
        """
        Concatenates and returns all text extracted from the PDF document.
        
        :return: A string containing all text from the document.
        """
        self.all_text = ""
        for page, text in self.pages.items():
            self.all_text += text
        return self.all_text
    
    def create_chunks(self):
        """
        Splits the entire text of the PDF into smaller chunks, useful for processing large texts or for input into NLP models.
        
        :return: A list of text chunks.
        """
        text = self.get_all_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,   # The size of each chunk in characters.
            chunk_overlap=200, # The overlap between consecutive chunks.
            length_function=len
        )
        self.chunks = text_splitter.split_text(text=text)
        print(f"chunks created, number of chunks: {len(self.chunks)}")
        return self.chunks

    @staticmethod
    def from_local_file(file_path):
        """
        Static method to load a PDF from a local file path into a PDFWrapper instance.
        
        :param file_path: Path to the PDF file.
        :return: A new instance of PDFWrapper.
        """
        with open(file_path, "rb") as f:
            local_file_bytes = io.BytesIO(f.read())
            reader = PyPDF2.PdfReader(local_file_bytes)
            return PDFWrapper(reader)
    
    @staticmethod
    def from_byte_stream(byte_stream):
        """
        Static method to initialize a PDFWrapper instance from a byte stream, commonly used with web applications.
        
        :param byte_stream: A byte stream of the PDF file, typically from an upload interface.
        :return: A new instance of PDFWrapper.
        """
        reader = PyPDF2.PdfReader(byte_stream)
        return PDFWrapper(reader)
    