import pickle
import re
import os
from pdf_wrapper import PDFWrapper
import llm_wrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI


class PaperDigest:
    """
    A class for processing and analyzing PDF documents by digesting them into manageable components.
    This includes summarizing text, creating embeddings, and storing vectors for efficient retrieval and analysis.

    The Initialization step in RAG
    """

    def __init__(self, pdf_wrapper: PDFWrapper):
        """
        Initializes the PaperDigest with a PDF wrapper instance to manage the underlying PDF operations.
        
        :param pdf_wrapper: An instance of PDFWrapper to handle PDF operations.
        """
        self.pdf_wrapper = pdf_wrapper
        self.vector_store = self.digest()
    
    def digest(self):
        """
        Processes the PDF file by embedding its textual content into a vector space model. This allows for 
        efficient similarity searches and retrieval operations.
        
        :return: An instance of a vector store, typically FAISS, containing the embedded vectors.
        """
        print('[API Call] Embedding')
        # initialize embedding odel
        embeddings = OpenAIEmbeddings(
                                      model='text-embedding-3-small')
        # split into chunks
        chunks = self.pdf_wrapper.create_chunks()
        # create a vector store (chunk -> embedding -> vector store)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        return VectorStore
    
    def save_to_local(self, store_path: str):
        """
        Saves the vector store to a local file for persistence and later retrieval.
        
        :param store_path: The file path where the vector store will be saved.
        """
        with open(store_path, "wb") as f:
            pickle.dump(self, f)
    
    def get_best_doc(self,query:str,top_k=1):
        """
        Performs a similarity search to find and return the most relevant document chunks based on the input query.
        Retrieval in RAG
        
        :param query: A string query for which the most similar document chunks are to be retrieved.
        :param top_k: int, get top_k results
        :return: A list of the top 3 most similar document chunks.
        """
        docs = self.vector_store.similarity_search(query=query, k=top_k) # get top_k most similar docs
        return docs

    def rerank_docs(self, reranker_model, query, retrieved_docs, top_k=1, output_score=False):
        """
        Rerank retrieved documents based on their relevance to a given query using a CrossEncoder model.

        Parameters:
        reranker_model: a BERT style model for reranking
        query (str): The search query to compare against the retrieved documents.
        retrieved_docs (list): A list of documents
        top_k (int, optional): The number of top-scoring documents to return. If None, returns all documents.
                            If specified, must be a positive integer.
        output_score (bool): If True, returns a tuple of two lists (documents, scores); if False, returns only a list of documents.

        Returns:
        list or tuple: Depending on the value of output_score:
                    - If output_score is False: Returns a list of documents sorted by descending score.
                    - If output_score is True: Returns a tuple containing two lists, one of the documents and one of their corresponding scores,
                                                both sorted by descending score.
        """
        # Create tuples of (query, document content) for scoring
        query_and_docs = [(query, r) for r in retrieved_docs]
        
        # Get scores for each document
        scores = reranker_model.predict(query_and_docs)
        
        # Sort documents by score in descending order
        sorted_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Apply top_k if it is specified and less than the length of the docs
        if top_k is not None and top_k < len(sorted_docs):
            sorted_docs = sorted_docs[:top_k]
        
        if output_score:
            # Separate the documents and their scores into two lists
            docs, scores = zip(*sorted_docs)
            return list(docs), list(scores)
        else:
            # Return only the documents
            docs = [doc for doc, score in sorted_docs]
            return docs

    @staticmethod
    def load_from_local(store_path: str):
        """
        Loads a saved vector store from a local file.
        
        :param store_path: The file path where the vector store is saved.
        :return: The loaded vector store if the file exists.
        :raises FileNotFoundError: If the specified file does not exist.
        """
        if os.path.exists(store_path):
            with open(store_path, "rb") as f:
                vectorstore = pickle.load(f)
            return vectorstore
        else:
            raise FileNotFoundError(f"No file found at specified path: {store_path}")
            
class CuratedPaper:
    TEMPLATE_SUMMARY = "Please summarize this section of a research paper. Be focused on the information a researcher may find interesting. Include the authors of the paper if you find them. The content you need to summarize is as below. {content}"
    TEMPLATE_INTRO = "Here are the summarizes of each page from a research paper. Please give yourself a proper name given the context of this paper, greet me as this paper, and introduce yourself to me with a summary of the content. Please also provide three questions a researcher may want to ask you like this 'You may want to ask me these questions...<bullet points>'. You don't need to include the original summaries in the response. The summaries are. {all_summaries}"
    TEMPLATE_QUESTION_GET_PAGE = 'Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. Context: "{all_summaries}" Question: Which page may best answer the question? Please just give me the page number in digits. The question is: {question}'
    TEMPLATE_ANSWER_WITH_PAGE = 'Answer the question as if you are a research paper based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. Context: "{content}" Question: {question}'
    TEMPLATE_ANSWER_WITH_SUMMARY = 'Answer the question as if you are a research paper based on the context below. Context: "{all_summaries}" Question: {question}'

    MAX_SUMMARY_TOKEN_COUNT = 4000

    def __init__(self, pdf_wrapper: PDFWrapper):
        self.version = VERSION
        self.pdf_wrapper = pdf_wrapper
        self.page_summary_map = {}
        self.per_page_limit = math.floor(
            self.MAX_SUMMARY_TOKEN_COUNT / pdf_wrapper.get_num_pages()
        )
        self.summary_all = self.curate_summary_all()

    def get_summary_for_page(self, page_num: int):
        if page_num in self.page_summary_map:
            return self.page_summary_map[page_num]
        page_text = self.pdf_wrapper.get_page(page_num)
        prompt = self.TEMPLATE_SUMMARY.format(content=page_text)
        answer = LLMWrapper.ask(
            prompt=prompt,
            max_tokens=self.per_page_limit,
        )
        return answer

    def curate_summary_all(self):
        num_pages = self.pdf_wrapper.get_num_pages()
        summaries = []

        with alive_bar(num_pages, length=40, spinner="dots_waves") as bar:
            for p in range(num_pages):
                summary = self.get_summary_for_page(p)
                summary_with_page = f"Page {p+1}: {summary}"
                summaries.append(summary_with_page)
                bar()

        summary_all = "\n\n".join(summaries)
        return summary_all

    def get_best_page_for_answer(self, question: str):
        prompt = self.TEMPLATE_QUESTION_GET_PAGE.format(
            question=question, all_summaries=self.summary_all
        )
        answer = LLMWrapper.ask(prompt=prompt, max_tokens=10)
        page_num = re.search(r"\d+", answer)
        if page_num is None:
            return None
        return int(page_num.group())

    def get_answer_from_page(self, question: str, page_num: int):
        page_text = self.pdf_wrapper.get_page(page_num - 1)
        prompt = self.TEMPLATE_ANSWER_WITH_PAGE.format(
            question=question,
            content=page_text,
        )
        answer = LLMWrapper.ask(prompt=prompt)
        answer += f" (Page {page_num})"
        return answer

    def get_answer_from_summary(self, question: str):
        prompt = self.TEMPLATE_ANSWER_WITH_SUMMARY.format(
            question=question, all_summaries=self.summary_all
        )
        answer = LLMWrapper.ask(prompt=prompt)
        return answer

    def get_answer_full_process(self, question: str):
        best_page = self.get_best_page_for_answer(question)
        if best_page is None:
            answer = self.get_answer_from_summary(question)
        else:
            answer = self.get_answer_from_page(question, best_page)
        return answer

    def get_intro(self):
        prompt = self.TEMPLATE_INTRO.format(all_summaries=self.summary_all)
        answer = LLMWrapper.ask(prompt=prompt)
        return answer

    def save_to_local(self, path: str):
        # save as binary
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_cache(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
