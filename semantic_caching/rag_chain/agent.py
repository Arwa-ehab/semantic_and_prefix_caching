
import os

from langchain_oci import OCIGenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomVectorizer
from langchain_community.vectorstores import FAISS
from langdetect import detect
from rag_chain.prompts import CONDENSE_QUESTION_PROMPT_EN, CONDENSE_QUESTION_PROMPT_AR, ANSWER_PROMPT_EN, ANSWER_PROMPT_AR
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
_runnable_print = RunnableParallel(
    {
        "print":RunnableLambda(lambda x: print(x)),
        "pass": RunnablePassthrough()
    }
) | RunnableLambda (lambda x: x['pass'])

# detect language function
def _detect_language(text):
    try:
        language = detect(text)
        return "Arabic" if language == 'ar' or language == 'fa' or language == 'ur' or language == 'sd' else "English"
    except:
        return "English"
    
def _is_lang_en(text):
    lang = _detect_language(text)
    return lang == "English" 

###########
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 

vectorstore_en = FAISS.load_local("semantic_and_prefix_caching\\semantic_caching\\rag_chain\\vector_data\\faiss_index_en_pdf", embeddings, allow_dangerous_deserialization=True)
vectorstore_ar = FAISS.load_local("semantic_and_prefix_caching\\semantic_caching\\rag_chain\\vector_data\\faiss_index_ar_pdf", embeddings, allow_dangerous_deserialization=True)


# # Create a retriever from the vector store
retriever_en = vectorstore_en.as_retriever()
retriever_ar = vectorstore_ar.as_retriever()


# # Create a RunnableBranch for retriever 
retriever =  RunnableBranch(
    # If the input is English
    (
        RunnableLambda(lambda x: _is_lang_en(x)),
        retriever_en
    ),
    # Else (Arabic)
    retriever_ar
).with_config({"run_name": "retrieverChain"})





# Define a function to combine multiple documents into a single string
def _combine_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

_search_query = RunnableBranch(
    # If input includes chat_history, condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: x["chat_history"],
            question = lambda x: x["question"]
        )
        | RunnableLambda( lambda x: CONDENSE_QUESTION_PROMPT_EN if x['is_english'] else CONDENSE_QUESTION_PROMPT_AR)
        | ChatOpenAI(model="gpt-4.1",temperature=0, verbose=True)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
).with_config({"run_name": "searchQueryChain"})

cache_embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=os.getenv("SERVICE_ENDPOINT"),
    compartment_id=os.getenv("COMPARTMENT_ID"),
)  
vectorizer=CustomVectorizer(cache_embeddings.embed_query)  # Use the same embeddings for vectorization
cache = SemanticCache(
    name="redis-server",
    redis_url="redis://localhost:6379",
    distance_threshold=0.8, # Your similarity threshold
    vectorizer=vectorizer
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"] ,
        "is_english": lambda x: _is_lang_en(x["question"]),
        "chat_history": lambda x: x["chat_history"],
    }
)
# -----------------------------
def check_cache(x):

    cached_results = cache.check(
        prompt=x["question"],
        distance_threshold=0.4
    )

    if len(cached_results) > 0:
        print("Using cached response")

        cached_response = cached_results[0]['response']

        return {
            **x,
            "cached": True,
            "response": cached_response
        }

    return {
        **x,
        "cached": False
    }


def return_cached(x):
    return {"content": x["response"]}


# -----------------------------
# Router
# -----------------------------
def route_cache(x):

    if x["cached"]:
        return RunnableLambda(return_cached)

    return continue_chain

# Define the final chain of runnables for answer synthesis
context_chain = (
    _search_query
    | retriever
    | _combine_documents
).with_config({"run_name": "contextChain"})

_gen_chain = (RunnableLambda(lambda x: ANSWER_PROMPT_EN if x['is_english'] else ANSWER_PROMPT_AR) | ChatOpenAI(model="gpt-4.1", temperature=0,streaming=True)).with_config({"run_name": "generationChain"})
continue_chain = (
    RunnablePassthrough.assign(
        context=context_chain
    )
    | _gen_chain
)

chain = (
    _inputs.assign(question=_search_query)
    | RunnableLambda(check_cache).with_config({"run_name": "cacheCheck"})
    | RunnableLambda(route_cache).with_config({"run_name": "cacheRoute"})
)


