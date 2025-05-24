from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import os
from dotenv import load_dotenv

from models.promptRequestModel import PromptRequest

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

output_parser = StrOutputParser()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.1,
    google_api_key=API_KEY
)

app = FastAPI()

@app.post("/getJoke")
async def getJoke(request: PromptRequest):
    response = llm.invoke("Tell me a Joke")
    return JSONResponse(content={"response": str(response.content)})