import os

from langchain_oci import OCIGenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomVectorizer
from dotenv import load_dotenv
from rag_chain.agent import chain
load_dotenv()  # Load environment variables from .env file

faq_questions = [
    "What is the maximum number of actual working hours permitted per week?",
    "How do working hours change during the month of Ramadan?",
    "Are there exceptions to the standard working hours for employees with special needs?",
    "How long is the daily rest period, and is it paid?",
    "Can I use my rest period to arrive late or leave work early?",
    "What is the breastfeeding break entitlement for female employees?",
    "Which employees are eligible for overtime compensation?",
    "What is the pay rate for overtime hours?",
    "Is there a limit on how many overtime hours I can work per month?",
    "Do employees receiving an 'On Call' allowance get overtime pay?",

    "What happens if an official holiday falls on a Friday?",
    "Am I compensated if I am required to work during Eid or other official holidays?",
    "What is the specific date and policy for the National Day holiday?",
    "Can I carry over my unused annual leave to the next year?",
    "Is it possible to work for another company during my annual leave?",
    "What is the pay structure for sick leave over the course of a year?",
    "What are the requirements for taking leave to attend academic exams?",
    "How many days of paid leave are granted for marriage or a newborn baby?",
    "What is the duration and eligibility for Hajj leave?",
    "What are the maternity leave entitlements for pregnant employees?",

    "How does the company define workplace bullying?",
    "Is harassment occurring outside of the physical workplace covered by policy?",
    "What are the dress code requirements for male employees?",
    "What are the dress code requirements for female employees?",
    "Can I be harassed if I do not comply with the dress code?",
    "What is the 'Confidentiality Protocol' regarding harassment complaints?",
    "What are the specific capacity rules for using elevators?",
    "Are employees allowed to eat at their desks?",
    "What is the policy on smoking within company facilities?",
    "Can I enter the workplace outside of official working hours?"
]
#Intialize the embeddings and cache

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=os.getenv("SERVICE_ENDPOINT"),
    compartment_id=os.getenv("COMPARTMENT_ID"),
)  
vectorizer=CustomVectorizer(embeddings.embed_query)  # Use the same embeddings for vectorization
cache = SemanticCache(
    name="redis-server",
    redis_url="redis://localhost:6379",
    vectorizer=vectorizer
)
# Insert the faq questions and answers into the cache
for question in faq_questions:
    answer = chain.invoke({"question": question,"chat_history": []}).content
    print(f"Question: {question}")
    print(f"Answer (from chain): {answer}\n")
    cache.store(prompt=question, response=answer)
    
