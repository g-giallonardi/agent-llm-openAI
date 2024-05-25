import os
import re
from dotenv import load_dotenv
from typing import List, Union

from pymongo import MongoClient
from langchain.chains.llm import LLMChain
from pydantic.v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


ATLAS_DB_USER = os.getenv('ATLAS_DB_USER')
ATLAS_DB_PASSWORD = os.getenv('ATLAS_DB_PASSWORD')
ATLAS_HOSTNAME = os.getenv('ATLAS_HOSTNAME')
ATLAS_DB_NAME = os.getenv('ATLAS_DB_NAME')
ATLAS_APP_NAME = os.getenv('ATLAS_APP_NAME')



class GetCustomerFullNameInput(BaseModel):
    """
    Pydantic arguments schema for get_customer_full_name method
    """
    first_name: str = Field(..., description="The first name of the customer")


class GetCustomerEmailInput(BaseModel):
    """
    Pydantic arguments schema for get_customer_email method
    """
    full_name: str = Field(..., description="The full name of the customer")

class GetShopOpeningsInput(BaseModel):
    """
    Pydantic arguments schema for get_customer_email method
    """
    country: str = Field(..., description="The living country of the customer")

class GetPositionBySkillsInput(BaseModel):
    skill: list = Field(..., description="the skill requested ")

def get_customer_full_name(first_name: str) -> str:
    """
    Retrieve customer's full name given the customer first name.

    Args:
        first_name (str): The first name of the customer.

    Returns:
        str: The full name of the customer.
    """
    full_name = first_name + " Smith"
    return full_name


def get_customer_email(full_name: str) -> str:
    """
    Retrieve customer email given the full name of the customer.

    Args:
        full_name (str): The full name of the customer.

    Returns:
        str: The email of the customer.
    """
    email = f'{full_name.lower().replace(" ",".")}@gmail.com'
    print(email)
    return email


def get_shop_openings(country: str) -> str:
    """
    Retrieve customer email given the full name of the customer.

    Args:
        full_name (str): The full name of the customer.

    Returns:
        str: The email of the customer.
    """

    hours_opened = []

    if country.lower() == 'france':
        hours_opened = ['7:34AM', '7:48PM']
    if country.lower() == 'belgium':
        hours_opened = ['8:35AM', '4:58PM']
    return hours_opened


def get_position_by_skills(skill: str) -> str:
    client = MongoClient(f'mongodb+srv://{ATLAS_DB_USER}:{ATLAS_DB_PASSWORD}@{ATLAS_HOSTNAME}/{ATLAS_DB_NAME}')
    db = client.resume
    print(skill)

    positions = []
    for job in db.jobs.find({'skills': skill.lower()}):
        positions.append(f'{job["position"]} at {job["company"]}')
    result = ','.join(positions)
    print(result)
    return 'Here the position I had that required the python skill : Developer fullstack at google from 2019 to 2020'


# Initialize the LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
)

# Initialize the tools
tools = [
    StructuredTool.from_function(
        func=get_customer_full_name,
        args_schema=GetCustomerFullNameInput,
        description="Function to get customer full name.",
    ),
    StructuredTool.from_function(
        func=get_customer_email,
        args_schema=GetCustomerEmailInput,
        description="Function to get customer email",
    ),
    StructuredTool.from_function(
        func=get_shop_openings,
        args_schema=GetShopOpeningsInput,
        description="Function to get opening hour of the shop",
    ),
    StructuredTool.from_function(
        func=get_position_by_skills,
        args_schema=GetPositionBySkillsInput,
        description="Function to get the all positions you did with a speficied skill needed",
    )
]
print('VVVVVVVVVVVVVVVVV')
get_position_by_skills('JavaScript')
print('^^^^^^^^^^^^')
template = """
You are my avatar and here some infos about me:
- name : GIALLONARDI
- firstname : Guillaume
- job : Fullstack developer
- prefered color : orange

You have access to the following tools:
{tools}
Don't invent response and follow the exact return of the tools

You act like you talk to a tech recruiter.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the EXACT return of the tool used
Final Answer: the final answer to the original input question

Begin!

Question: {input}
"""

class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        print(kwargs)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

llm = ChatOpenAI(temperature=0.5,model_name="gpt-3.5-turbo",)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    # We use "Observation" as our stop sequence so it will stop when it receives Tool output
    # If you change your prompt template you'll need to adjust this as well
     stop=[],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent_executor.run("Are you already did python in past ? if yes what the company you worked for?")