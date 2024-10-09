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
        self.store = {}
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

        self.system_prompt_rag_log = (
        """
        Atue com a personalidade de um agente especialista em perguntas e respostas as bolsas (salários) dos funcionários da instituição CEIA.

        Considere que você está inserido no evento Conecta CEIA e os usuários querem saber sobre os salários dos funcionários.
        Responda de forma amigável.
        {context}
        """
        )

        self.contextualize_q_system_prompt = (
        """
        Atue com a personalidade de um agente especialista em perguntas e respostas as bolsas (salários) dos funcionários da instituição CEIA.
        Dada a história do chat e a última pergunta do usuário, que pode referenciar o contexto na história do chat, reformule a pergunta de forma que possa ser entendida sem a necessidade da história do chat.
        NÃO responda à pergunta, apenas reformule-a se necessário e, caso contrário, retorne-a como está.
        """
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def create_prompt(self, context_text, history):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{user_input}"),
            ]
        ).partial(context=context_text, history=history)
        prompt_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        return prompt_chain
    
    def create_log_prompt(self, context_text, history, source_type):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_log),
                ("human", "{user_input}"),
            ]
        ).partial(context=context_text, history=history, source_type=source_type)
        prompt_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        return prompt_chain
    
    def rag_history_prompt(self,context_text, message):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_rag_log),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, context_text, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=self.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )

        result = conversational_rag_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": "abc321"}},
        )["answer"]

        return result

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

        print("context_docs", context_docs)

        # Seleção de documentos caso tenha rag_tables
        filtered_docs = [
            doc for doc in context_docs if doc['score'] >= 0.55
        ] if rag_tables else []

        context_text = "\n".join([doc['text'] for doc in filtered_docs]) if rag_tables else ''


        history_formated = self.format_history(history) if history else ''

        # Seleção de prompt com base no modo log

        if log and rag_tables:
            result = self.rag_history_prompt(context_text, message)
            return result

        elif log and rag_video:
            #Falta o rag de vídeo 
            prompt_chain = ""

        elif rag_video:
            #Falta o rag de vídeo 
            prompt_chain = ""

        elif log:
           prompt_chain = self.create_log_prompt(context_text, history_formated, source_type)

        elif rag_tables:
            prompt_chain = self.create_prompt(context_text, history_formated)
            
        else:
            prompt_chain = self.create_log_prompt(context_text, history_formated, source_type)


        result = prompt_chain.invoke({"user_input": message})

        return result
