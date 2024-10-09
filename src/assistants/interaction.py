from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.utils.vector_search import mongodb_vector_search

import os

class Assistant:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt_log = """
            Você é o assistente personalizado do evento Conecta CEIA.
            Você está configurado no modo logger.
            Não responda NADA NUNCA sobre perguntas que não são sobre o conecta CEIA

            Informe em formato de lista sobre:

            0. O usuário perguntou sobre:
            {user_input}

            1. Histórico da conversa
            {history}

            2. Documentos de RAG encontrados:
            {context}

            3. Tipo da fonte:
            {source_type}

            4. A resposta para a pergunta.
            
            **IMPORTANTE***
            A resposta deve ser rica em detalhes sem erros de português.
            Não responda NADA NUNCA sobre perguntas que não são sobre o conecta CEIA
        """

        self.system_prompt = """
            Você é o assistente personalizado do evento Conecta CEIA e está conversando com o usuário.
            Não responda NADA NUNCA sobre perguntas que não são sobre o conecta CEIA

            Histórico da conversa:
            {history}

            O usuário perguntou sobre: {user_input}
            Valor retornado: {context}

            Para responder considere:

            1. Se não souber responda que não sabe.
            2. Não fale sobre algo que não está no valor retornado.
            3. Não invente valores.
            4. Não responda NADA NUNCA sobre perguntas que não são sobre o conecta CEIA

        """

    def create_prompt(self, context_text, history):
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{user_input}"),
            ]
        ).partial(context=context_text, history=history)
    
    def create_log_prompt(self, context_text, history, source_type):
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_log),
                ("human", "{user_input}"),
            ]
        ).partial(context=context_text, history=history, source_type=source_type)

    def format_history(self, history_list):
        return "\n".join(
            "{}: {}".format(history['role'], history['content']) 
            for history in history_list
        )
    
    def rag_chain(self, message, history='', rag_tables=False, rag_video=False, log=False, source_type='csv'):
        
        vector_index = os.getenv("CONNECTA_VECTOR_INDEX_NAME")

        context_docs = mongodb_vector_search(
            vector_index=vector_index,
            text=message,
            context="connecta ceia",
            k=1
        ) if rag_tables else []

        # Seleção de documentos caso tenha rag_tables
        filtered_docs = [
            doc for doc in context_docs if doc['score'] >= 0.55
        ] if rag_tables else []

        context_text = "\n".join([doc['text'] for doc in filtered_docs]) if rag_tables else ''

        history_formated = self.format_history(history) if history else ''

        # Seleção de prompt com base no modo log

        if log:
            prompt = self.create_log_prompt(context_text, history_formated, source_type)

        elif rag_tables:
            prompt = self.create_prompt(context_text, history_formated)

        else:
            prompt = self.create_prompt(context_text, history_formated)

        print(prompt)

        rag_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        result = rag_chain.invoke({"user_input": message})

        return result
