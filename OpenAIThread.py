import time
import subprocess
import textwrap
import builtins

from typing import Callable, Literal, Optional
from typing import List
from pydantic import Field

from dotenv import load_dotenv
import openai
from openai.types.beta.assistant_create_params import ToolAssistantToolsFunction
from openai.types.shared_params.function_definition import FunctionDefinition

import instructor
from instructor import OpenAISchema


print(openai.__version__)
load_dotenv()
client = openai.Client()


def wprint(*args, width=70, **kwargs):
    """
    Custom print function that wraps text to a specified width.

    Args:
    *args: Variable length argument list.
    width (int): The maximum width of wrapped lines.
    **kwargs: Arbitrary keyword arguments.
    """
    wrapper = textwrap.TextWrapper(width=width)

    # Process all arguments to make sure they are strings and wrap them
    wrapped_args = [wrapper.fill(str(arg)) for arg in args]

    # Call the built-in print function with the wrapped text
    builtins.print(*wrapped_args, **kwargs)


def get_completion(message, agent, funcs, thread):
    """
    Executes a thread based on a provided message and retrieves the completion result.

    This function submits a message to a specified thread, triggering the execution of an array of functions
    defined within a func parameter. Each function in the array must implement a `run()` method that returns the outputs.

    Parameters:
    - message (str): The input message to be processed.
    - agent (OpenAI Assistant): The agent instance that will process the message.
    - funcs (list): A list of function objects, defined with the instructor library.
    - thread (Thread): The OpenAI Assistants API thread responsible for managing the execution flow.

    Returns:
    - str: The completion output as a string, obtained from the agent following the execution of input message and functions.
    """

    # create new message in the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )

    # run this thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
    )

    while True:
        # wait until run completes
        while run.status in ['queued', 'in_progress']:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            time.sleep(1)

        # function execution
        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                wprint('\033[31m' + str(tool_call.function), '\033[0m')
                # find the tool to be executed
                func = next(
                    iter([func for func in funcs if func.__name__ == tool_call.function.name]))

                try:
                    # init tool
                    func = func(**eval(tool_call.function.arguments))
                    # get outputs from the tool
                    output = func.run()
                except Exception as e:
                    output = "Error: " + str(e)

                wprint(f"\033[33m{tool_call.function.name}: ",
                       output, '\033[0m')
                tool_outputs.append(
                    {"tool_call_id": tool_call.id, "output": output})

            # submit tool outputs
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        # error
        elif run.status == "failed":
            raise Exception("Run Failed. Error: ", run.last_error)
        # return assistant message
        else:
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            message = messages.data[0].content[0].text.value
            return message

# code assistant ----------------------------------------------------------------------------

# # functions
# class ExecutePyFile(OpenAISchema):
#     """Run existing python file from local disc."""
#     file_name: str = Field(
#         ..., description="The path to the .py file to be executed."
#     )

#     def run(self):
#       """Executes a Python script at the given file path and captures its output and errors."""
#       try:
#           result = subprocess.run(
#               ['python3', self.file_name],
#               text=True,
#               capture_output=True,
#               check=True
#           )
#           return result.stdout
#       except subprocess.CalledProcessError as e:
#           return f"An error occurred: {e.stderr}"
# class File(OpenAISchema):
#     """
#     Python file with an appropriate name, containing code that can be saved and executed locally at a later time. This environment has access to all standard Python packages and the internet.
#     """
#     chain_of_thought: str = Field(...,
#         description="Think step by step to determine the correct actions that are needed to be taken in order to complete the task.")
#     file_name: str = Field(
#         ..., description="The name of the file including the extension"
#     )
#     body: str = Field(..., description="Correct contents of a file")

#     def run(self):
#         with open(self.file_name, "w") as f:
#             f.write(self.body)

#         return "File written to " + self.file_name

# code_assistant_funcs = [File, ExecutePyFile]

# # code assistant defination
# code_assistant = client.beta.assistants.create(
#   name='Code Assistant Agent',
#   instructions="As a top-tier programming AI, you are adept at creating accurate Python scripts. You will properly name files and craft precise Python code with the appropriate imports to fulfill the user's request. Ensure to execute the necessary code before responding to the user.",
#   model="gpt-4-1106-preview",
#   tools=[{"type": "function", "function": File.openai_schema},
#          {"type": "function", "function": ExecutePyFile.openai_schema}]
# )


# lead qualifyer ----------------------------------------------------------------------------
lead_qualifier = client.beta.assistants.create(
    name='Qualifying Agent',
    instructions="""You are a Qualifying agent based on the following crieteria, let AVA know what is the next qualifying question to ask the user:
    Q1: are you the homeowner? Acceptable answer: yes

    Q2: how old is your roof? acceptable answer: less than 15 years old
    -> if the answer to Q2 is more than 15 years old: ask if there are any leaks. Acceptable answer: no
    -> if the answer to Q2 is more than 15 years old and there are leaks: ask if the user is interested in a roof repair. Acceptable answer: yes

    Q3: do you have a credit score of 650 or higher? Acceptable answer: yes

    Note: you ask ONLY one question at a time.
    """,
    model="gpt-4-1106-preview",
    tools=[]
)

# appointment setter ----------------------------------------------------------------------------
# functions
class BookAppointment(OpenAISchema):
    """Book an appointment for the user."""
    phone: str = Field(...,
                       description="The phone number of the user.")
    email: str = Field(...,
                       description="The email of the user.")
    start_time: str = Field(...,
                            description="The start time of the appointment.")
    time_zone: str = Field(...,
                           description="The time zone of the user.")
    meeting_title: str = Field(...,
                               description="The title of the meeting.")

    def run(self):
        """Book an appointment for the user."""
        print(self.phone, self.email, self.start_time,
              self.time_zone, self.meeting_title)
        return "Appointment booked successfully."

class GetAppointmentSlots(OpenAISchema):
    """Get available appointment slots for the user."""
    phone: str = Field(...,
                       description="The phone number of the user.")
    email: str = Field(...,
                       description="The email of the user.")
    time_zone: str = Field(...,
                           description="The time zone of the user.")


    def run(self):
        """Get available appointment slots for the user."""
        print(self.phone, self.email, self.time_zone)
        return "Here are some available appointment slots. 10:00am today, 11:00am tomorrow, 1:00pm tomorrow."

appointment_setter = client.beta.assistants.create(
    name='Appointment Setter Agent',
    instructions=" As an appointment setter, you are responsible for booking appointments for the AVA. You will need to retrieve available appointment slots and book an appointment for the AVA.",
    model="gpt-4-1106-preview",
    tools=[ToolAssistantToolsFunction(type="function", function=FunctionDefinition(**GetAppointmentSlots.openai_schema)),
           ToolAssistantToolsFunction(type="function", function=FunctionDefinition(**BookAppointment.openai_schema))]
)



agents_and_threads = {

    # "code_assistant": {
    #     "agent": code_assistant,
    #     "thread": None,
    #     "funcs": code_assistant_funcs
    # },

    "lead_qualifier": {
        "agent": lead_qualifier,
        "thread": None,
        "funcs": []
    },

    "appointment_setter": {
        "agent": appointment_setter,
        "thread": None,
        "funcs": [GetAppointmentSlots, BookAppointment]
    }
}

class SendMessage(OpenAISchema):
    """Send messages to other specialized agents in this group chat."""
    recepient: Literal['lead_qualifier', 'appointment_setter'] = Field(...,
                                                                       description="""You have the following agents that may be helpfull
                      1. lead_qualifier: it is a world-class qualifier who helps qualifying the user for the service provided by the business
                      2. appointment_setter: it is a world-class appointment setter who helps booking appointments""")
    message: str = Field(...,
                         description="Specify the task required for the recipient agent to complete. Focus on clarifying what the task entails, rather than providing detailed instructions.")

    def run(self):
        recepient = agents_and_threads[self.recepient]
        # if there is no thread between user proxy and this agent, create one
        if not recepient["thread"]:
            recepient["thread"] = client.beta.threads.create()

        message = get_completion(message=self.message, **recepient)

        return message

ava_tools = [SendMessage]

ava = client.beta.assistants.create(
    name='Ava',
    instructions="""
  
  # Discription
  You are "Ava", and your job is to provide support to my clients. 
  You are a solar energy assistatn working for the lovely "Solar Queen Fina" aka Fina.
  
  Your duty is to articulate the user message accurately and use various capability you possess, along with the help of other agents, to provide the best possible response to the user.

  # Instruction
  You are an Artificial Virtual Assistant (AVA) that is responsible for communicating with users and delegating tasks to specialized agents.
  You are a generalist, an "intermediary communication" agent, and with your help, users can communicate with specialized agents.
  You maintain a natural conversation flow with the user. 
  The user is not aware of the specialized agents and only communicates with you.
  
  
""",
    model="gpt-4-1106-preview",
    tools=[
        ToolAssistantToolsFunction(type="function", function=FunctionDefinition(**SendMessage.openai_schema)),
        {"type": "code_interpreter"}
    ],
)

thread = client.beta.threads.create()

while True:
    user_message = input("User: ")

    message = get_completion(user_message, ava, ava_tools, thread)

    wprint(f"\033[34m{ava.name}: ", message, '\033[0m')

    with open("thread.json", "a") as f:
        f.write(client.beta.threads.messages.list(thread_id=thread.id).model_dump_json(indent=2))

