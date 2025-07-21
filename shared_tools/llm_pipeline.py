from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LLMPipeline:
    def __init__(self, llm):
        self.llm = llm

    def create_pipeline(self, template: str, input_variables: list) -> LLMChain:
        """
        Creates an LLM pipeline with a given template and input variables.

        :param template: The prompt template.
        :param input_variables: The input variables for the template.
        :return: An LLMChain object.
        """
        prompt = PromptTemplate(template=template, input_variables=input_variables)
        return LLMChain(llm=self.llm, prompt=prompt)
