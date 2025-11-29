# Medium.py
# Build a chain that takes a topic and generates a haiku about it

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# 1. Define a prompt template for haiku generation
prompt = ChatPromptTemplate.from_template(
    """You are a poetic assistant.
Write a 3-line haiku about the topic below.

Topic: {topic}

Rules:
- Exactly 3 lines
- No line numbers or extra text
- Keep it simple and evocative"""
)

# 2. Create the LLM (ensure OPENAI_API_KEY is set in your environment)
llm = ChatOpenAI(
    model="gpt-5-nano",   # or "gpt-4o-mini" / any other chat model you use
    temperature=0.7        # a bit of creativity for poetry
)

# 3. Parser: get plain text from the LLM response
parser = StrOutputParser()


# 4. Build the LCEL chain: Prompt → LLM → Text
chain = prompt | llm | parser

# 5. Example topics to test the chain
topics = [
    "sunrise over the ocean",
    "machine learning",
    "autumn leaves",
    "quiet library",
    "vector databases"
]

# 6. Run the chain for each topic and print results
if __name__ == "__main__":
    for i, topic in enumerate(topics, start=1):
        print("\n" + "=" * 60)
        print(f"{i}. TOPIC: {topic.upper()}")
        print("=" * 60)

        haiku = chain.invoke({"topic": topic})
        print(haiku)
