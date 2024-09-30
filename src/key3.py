from groq import Groq
from src.api import KEY

client = Groq(
    api_key=KEY,
)

def create_queries(subject) :

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Generate 15 google scholar queries from this subject : \"{subject}\" Your response should only contain the queries, no title, no quotation marks, no numbers, one per line.",
            }
        ],
        model="llama3-8b-8192",
    )

    log = ""

    queries = chat_completion.choices[0].message.content.split("\n")

    return queries, log
