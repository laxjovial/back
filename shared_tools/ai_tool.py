import os
from openai import OpenAI

class AITool:
    def __init__(self, openai_api_key: str = None):
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or pass it to the constructor.")
        self.client = OpenAI(api_key=openai_api_key)

    def get_ai_insight(self, data: dict, prompt: str) -> str:
        """
        Leverages a large language model to provide insights on the given data.

        :param data: The data to be analyzed.
        :param prompt: The prompt to guide the AI's analysis.
        :return: The AI-generated insight.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides insights on data."},
                    {"role": "user", "content": f"{prompt}\n\nData:\n{data}"}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred while generating AI insight: {e}"
