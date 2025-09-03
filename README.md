# AI INTERVIEWER 

An AI-driven voice interviewer capable of conducting deep, long-form interviews (up to ~2 hours) and extracting the most meaningful information from the conversation.

> **Project status:** ongoing development  
> **Repository layout:**  
> - `BACK_END/` – FastAPI + NLP pipeline  
> - `FRONT_END/` – UI and client app

---
Hi! 
### Here you can see how the program has to be run the first time:
Python v 3.12 
1) Go to the directory of the back_end. 'cd ./BACK_END'
2) Install all the back_end requirements:  'pip install -r requirements.txt' 

3) Open a new terminal and go to the directory of the front_end. 'cd ./FRONT_END'
4) Install all the front_end requirements: 'setup.sh' or './setup.sh'
5) To start the front end: 'start.sh' or './start.sh'

6) To start the back_end go to the back_end terminal. 'python -m uvicorn Main.main:app --reload'
7) Create a file .env and then you have to write there the following personal key and setup:
- Open Ai key (
OPENAI_API_KEY=...,
), 
- AWS Polly (
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY= ... 
AWS_REGION=us-east-1
AWS_POLLY_VOICE_ID=Bianca
AWS_POLLY_ENGINE=neural  # neural o standard
AWS_POLLY_FORMAT=mp3     # mp3, ogg_vorbis, pcm
),
- development mode (
DEVELOPMENT_MODE=false 
),

- Tresholds NLP (
COVERAGE_THRESHOLD_PERCENT=90
TH_FUZZY=75
TH_COS=60
),

### To start the program in the next times:

1)Open two terminal. One for the back_end and oine for the front_end.
2)Go to the directories. (just like above)
3)back_end: 'python -m uvicorn Main.main:app --reload'
4)front_end: 'start.sh' or './start.sh'



### Output data:
        user_id: Identifier 
        session_id: session ID
        question_idx: Question index
        question_text: Question text
        response_text: User's answer text 
        topic: Topic of the question 
        subtopics: A list of the subtopics related to the topic 
        keywords: A dictionary of the keywords related to each subtopic 
        non_covered_subtopics: List of subtopics not covered by the user's answer. 
        coverage_percent: User's answer percent coverage related to the following rapport - (Number of subtopics detected in the answer (that are related to the topic) / Total number of the subtopics related to the topic )*100



### Licence
This code is published for demonstration purposes. All rights reserved.
No part of this repository may be copied, modified, or used without explicit written permission from the author.
