import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import time
from datetime import datetime

load_dotenv()

# Configure Gemini
