
from datetime import datetime, time

from rag_chain.agent import chain
#Testing rag with a question from the cache
start_time = datetime.now()
response=chain.invoke({"question": "What is the policy on smoking within company facilities?","chat_history": []})
end_time = datetime.now()
print("Testing question from cache:")
print(f"Time taken: {end_time - start_time} seconds")
#Testing questions that are not in the cache
start_time = datetime.now()
response_1=chain.invoke({"question": "I have been subjected to discrimination","chat_history": []})
end_time = datetime.now()
print("Testing question not in cache:")
print(f"Time taken: {end_time - start_time} seconds")
