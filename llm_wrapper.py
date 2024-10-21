import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI


class LLMWrapper:
    """
    A utility class for interfacing with OpenAI's language models. This class simplifies the process of generating responses and follow-up questions using the OpenAI API, specifically tailored for handling natural language processing tasks.
    """
    
    @staticmethod
    def init(api_key):

        pass
    TEMPLATE_QUESTION_GET_PAGE = ' Context: "{all_summaries}" Question: Which page may best answer the question? Please just give me the page number in digits. The question is: {question}'

    @staticmethod
    def rewrite_query(query):
        '''
        The query rewrite step in RAG to enhance retrieval performance.

        :param query: str, user input query
        :return: str, re-written query
        '''
        prompt_template = """You are an assistant specialized in enhancing search queries to improve retrieval performance. 
Given the user's query below, rewrite it to make it more informative and precise while retaining its original intent but keep it concise.

User Query: "{question}"
Rewritten Query:"""  # Template to guide the language model

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables="question"
        )

        # Initialize the language model
        llm = OpenAI(
            model_name="gpt-4o-mini",
            temperature=1.0,  # Larger temperature to provide some diversity
            top_p=1,  # Total probability mass of tokens to consider at each step
            max_tokens=200  # Max token limit for the output
        )

        # Load a specific QA chain setup for this task
        chain = LLMChain(llm=llm, prompt=PROMPT)
        print('[API Call - Query Rewrite] gpt-4-mini')  # Adjust log message according to actual model used
        
        # Execute the chain with the query and extract the output
        response = chain({"question": query}, return_only_outputs=True)
        # Extract and return the rewritten query from the response
        return response['text'].strip()
    
    @staticmethod
    def ask(query, docs):
        """
        Generates an answer to a given query based on the provided documents using a specified model from OpenAI's language model offerings.

        :param query: The user's question as a string.
        :param docs: The context or documents related to the question.
        :return: A string containing the model's response to the question.
        """
        prompt_template = """<im_start>system
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer, don't try to make up an answer.
<im_end>
<im_start>user
# Context:
{context}
# Question: {question}
<im_end>
<im_start>assistant
"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        llm = OpenAI(
                    model_name="gpt-4o-mini",
                     temperature=0, # What sampling temperature to use.
                     top_p = 1, # Total probability mass of tokens to consider at each step.
                     max_tokens = 800)
        chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)
        print('[API Call] gpt4o-mini')
        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return response['output_text']
    
    @staticmethod
    def generate_next_question(prev_query,prev_ans,docs):
        """
        Generates potential follow-up questions based on the context, a previous question, and its answer.

        :param prev_query: The previous question asked by the user.
        :param prev_ans: The answer provided to the previous question.
        :param docs: The context or documents the previous question and answer were based on.
        :return: A string containing three suggested follow-up questions.
        """
        prompt_template = """<im_start>system
A user asked a question: {question}, and you gave the answer: {ans} based on the context: {context}.
Please generate three possible questions the user may want to ask next.
Example output:
1. ...
2. ...
3. ...
<im_end>
<im_start>assistant
"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context","question","ans"]
        )
        llm = OpenAI(model_name="gpt-4o-mini",
                     temperature=1, # What sampling temperature to use.
                     top_p = 1, # Total probability mass of tokens to consider at each step.
                     max_tokens = 800)
        chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)
        print('[API Call] gpt4o-mini')
        response = chain({"input_documents": docs, "question":prev_query,"ans":prev_ans}, return_only_outputs=True)
        return response['output_text']


