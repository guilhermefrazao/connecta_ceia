from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


from src.utils.vector_search import mongodb_vector_search

import os

class Assistant:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt_log = """
            Você é o assistente personalizado do evento Conecta CEIA.
            Você está configurado no modo logger.
            Não responda NADA NUNCA sobre perguntas que não são sobre o conecta CEIA
            Contexto sobre o evento Conecta CEIA:

            "O Conecta CEIA é um evento anual focado na área de Inteligencia Artificial, promovido pelo Centro de Excellencia em Inteligencia Artificial do Estado de Goias.
            Sua sede fica situada na Universidade Federal de Goias(UFG), Campus Samambaia, Goiania - GO.
            Ele conta com o apoio de diversas empresas que investem em projetos na área de IA e Machine Learning.
            O evento é composto por palestras, workshops, minicursos e competições de IA.
            Sendo que a grande parte dos participantes são estudantes de graduação e pós-graduação da UFG e de outras instituições de ensino.
            Que fazem o curso de Inteligência Artificial, Ciência da Computação, Engenharia de Software, Engenharia de Computação, Sistemas de Informação, Matemática Computacional, Estatística e áreas afins."

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
            Você precisa informar sobre o que o sistema encontrou explicando pro usuário cada etapa.
            Não responda NADA NUNCA sobre perguntas que não são sobre o conecta CEIA
        """

        self.system_prompt = """
            Você é o assistente personalizado do evento Conecta CEIA e está conversando com o usuário.

            Histórico da conversa:
            {history}

            O usuário perguntou sobre: {user_input}
            Valor retornado: {context}

            Para responder considere:

            1. Se não souber responda que não sabe.
            2. Não fale sobre algo que não está no valor retornado.
            3. Não invente valores.
            4. Utilize o histórico da conversa se houver.
        """

    def create_prompt(self, context_text, history_formated):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{user_input}"),
            ]
        ).partial(context=context_text, history=history_formated)
        return prompt
    
    def create_log_prompt(self, context_text, history_formated, source_type):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_log),
                ("human", "{user_input}"),
            ]
        ).partial(context=context_text, history=history_formated, source_type=source_type)
        return prompt

    def format_history(self, history_list):
        return "\n".join(
            "Mensagem {}: {} - {}.".format(i+1, history['content'], history['role']) 
            for i, history in enumerate(history_list)
        )
    
    def rag_chain(
            self, message,  history={},
            rag_tables=False, log=False,
            source_type='csv', enable_history=False,
    ):
        
        print(f"Historico:{enable_history}, Logs:{log}, RAG:{rag_tables}")
        
        vector_index = os.getenv("CONNECTA_VECTOR_INDEX_NAME")

        print("rag_tables", rag_tables)

        if rag_tables:
            context_docs = mongodb_vector_search(
                vector_index=vector_index,
                text=message,
                context="connecta ceia",
                k=1
            )
        else:
            context_docs = []

        print("context: ", context_docs)

        # Seleção de documentos caso tenha rag_tables
        filtered_docs = [
            doc for doc in context_docs if doc['score'] >= 0.50
        ] if rag_tables else []

        context_text = "\n".join([doc['text'] for doc in filtered_docs]) if rag_tables else ''
        print(context_text)

        history_formated = self.format_history(history) if enable_history else ''
        print("HISTÓRICO", history_formated)

        prompt = self.create_prompt(context_text, history_formated) if not log \
            else self.create_log_prompt(context_text, history_formated, 'csv')

        print(str(prompt))

        prompt_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        result = prompt_chain.invoke({"user_input": message})

        return result
