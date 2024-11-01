from langchain.prompts import ChatPromptTemplate
from .helpers import get_chroma_db,PROMPT_TEMPLATE,MODEL_TEMP, GROQ_MODEL_ID
from langchain_groq import ChatGroq
from models import ResponseData
from dotenv import load_dotenv 

load_dotenv()

TOP_RETRIVAL_COUNT=5
RELEVANCE_TRESHOLD=0.6

def query_model(query_text: str) -> ResponseData:
    db = get_chroma_db()
    results = get_search_result(db,query_text)
    prompt = get_prompt(results,query_text)
    response = get_llm_response(prompt)
    final_response = format_response(response,results)
    # print(final_response)
    return final_response



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
        temperature=MODEL_TEMP
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
