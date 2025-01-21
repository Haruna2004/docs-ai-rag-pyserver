from langchain.prompts import ChatPromptTemplate
from src.sq_rag_app.helpers_sq import get_chroma_db,PROMPT_TEMPLATE,MODEL_TEMP, GROQ_MODEL_ID
from langchain_groq import ChatGroq
from models import ResponseData
from dotenv import load_dotenv 
import os

load_dotenv()

groq_key = os.getenv('GROQ_API_KEY')
TOP_RETRIVAL_COUNT=5
RELEVANCE_TRESHOLD=0.6


def query_model(query_text: str) -> ResponseData:
    try:
        db = get_chroma_db()
        results = get_search_result(db,query_text)
        if results == None:
            return no_result()
        prompt = get_prompt(results,query_text)
        response = get_llm_response(prompt)
        final_response = format_response(response,results)
        return final_response
    except Exception as e:
        return ResponseData(
       content=f"Error in query_model: {e}",
       sources=[],
       total_tokens=0
   )


def no_result():
   response = ResponseData(
       content="Sorry, I wasn't able to answer your question using the Paystack docs.",
       sources=[],
       total_tokens=0
   )
   return response


def get_search_result(db,query_text):
    results = db.similarity_search_with_relevance_scores(query_text, k=TOP_RETRIVAL_COUNT)
    if len(results) == 0 or results[0][1] < RELEVANCE_TRESHOLD:
        print(f"Unable to find matching results.")
        # Log the results
        return None
    return results


def get_prompt(results,query_text):
    # Retrival strategy
    context_text = "\n\n--\n\n".join([doc.page_content for doc,_score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text,question=query_text)
    return prompt


def get_llm_response(prompt):
    model = ChatGroq(
        model = GROQ_MODEL_ID,
        temperature=MODEL_TEMP,
        api_key=groq_key
    )
    print("Processing AI response..." )
    response = model.invoke(prompt)
    return response

def format_response(response, results) -> ResponseData:
    final_response = ResponseData(
        content=response.content,
        sources=[
        {'source_link': doc.metadata.get("source", None), 'relevance': score}
        for doc, score in results
        ],
        total_tokens= response.response_metadata['token_usage']['total_tokens']
    )
    return final_response


if __name__ == "__main__":
    query_model("how do I add payment to an android app?")
