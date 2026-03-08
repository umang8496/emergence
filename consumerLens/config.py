import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4o-mini"


# Check the key
if (not API_KEY) or (not isinstance(API_KEY, str)):
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif (not API_KEY.startswith("sk-proj-")):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif (API_KEY.strip() != API_KEY):
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")


client = OpenAI(api_key=API_KEY)
