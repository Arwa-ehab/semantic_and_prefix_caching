from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

condese_template_en = """ Given the following conversation and the follow-up question, rephrase the follow up question to be a standalone question only if it may be related to previous conversation. rephrase it using it's original language.
chat History :
{chat_history}
Follow Up Input: {question}
Standalone question:"""   # noqa: E501
CONDENSE_QUESTION_PROMPT_EN = PromptTemplate.from_template(condese_template_en)

######

condese_template_ar = """ Given the following conversation and the follow-up question, rephrase the follow up question to be a standalone question only if it may be related to previous conversation. rephrase it using it's original language.
chat History :
{chat_history}
Follow Up Input: {question}
Standalone question:"""  
CONDENSE_QUESTION_PROMPT_AR = PromptTemplate.from_template(condese_template_ar)

######

rag_template_en = """You are an assistant for a company called Petrolube having a conversation with a human about the documents retreived.\
    Use the following pieces of retrieved documents to answer the user question.\
    If the user question is considered as 'polite greetings', 'small talk', 'social niceties', or 'pleasantries', reply with a related answer to the conversation.\
    If the user question doesn't relate to Context retreived don't try to make up an answer just say 'I don’t have information on that topic. Please check with HR or the relevant department.'.\
    Formulate your response in clear points.\
       
    --------- 
    DOCUMENTS: 
    {context}
    ---------
    """
# Create a ChatPromptTemplate for answer synthesis
ANSWER_PROMPT_EN = ChatPromptTemplate.from_messages(
    [
        ("system", rag_template_en),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Define a template for RAG answer synthesis
rag_template_ar = """ أنت روبوت دردشة لشركه بترولوب تجري محادثة مع إنسان عن سياسات شركه بترولوب\
     بالنظر إلى الأجزاء المستخرجة التالية من مستندات ستجد إجابتك باللغة العربية، قم بإنشاء إجابة بنفس اللغة.\
      إذا كان السؤال لا يتعلق بالمستندات، ولك يعتبر "تحية مهذبة" أو "مجاملات اجتماعية"، قم بالرد بإجابة ذات صلة بالمحادثة عن شركه بترولوب\
    " أما إذا كان السؤال لا يتعلق بالمستندات بالاسفل و غير متعلق بمنتجات وخدمات  شركه بترولوب، فلا تحاول اختلاق إجابة فقط قل "لا تتوفر لدي معلومات حول هذا الموضوع. يُرجى التواصل مع إدارة الموارد البشرية أو الجهة المختصة."\
     قم بصياغة إجابتك في نقاط واضحة.\

    <المستندات>
    {context}
    </المستندات>
    """


    
    # Create a ChatPromptTemplate for answer synthesis
ANSWER_PROMPT_AR = ChatPromptTemplate.from_messages(
    [
        ("system", rag_template_ar),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)