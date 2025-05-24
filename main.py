
# * Import statements for langchains
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# * Import statements for FastAPI 
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# * Env file loading imports
import os
from dotenv import load_dotenv

# * models imports
from models.promptRequestModel import PromptRequest

# * Loads the env file to read the gemini api ket
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# * Initialize the FastAPI application
app = FastAPI()

output_parser = StrOutputParser()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.1,
    google_api_key=API_KEY
)

message = PromptTemplate(
    input_variables=['topic'],
    template="Give me a para about {topic}"
)

summary = PromptTemplate(
    input_variables=['input'],
    template="Give me a summary of {input} in two lines"
)

wrap_input = RunnableLambda(lambda x: {"input": x})

chain = (
    {"topic": RunnablePassthrough()}
    | message
    | llm
    | output_parser
    | wrap_input
    | summary
    | llm
    | output_parser
)

@app.post("/chatBot")
async def chatBot(request: PromptRequest):
    try:
        response = chain.invoke({"topic": request})
        return JSONResponse(content={"topic": request.prompt, "response": response})
    except Exception as error:
        raise HTTPException(status_code=500, detail=error)