import time
import requests
import json

CHATBOT_URL = "http://localhost:8000/movie-rag-agent"

questions = [
    "1. What is the overall sentiment analysis regarding the acting of Jane Fonda ? Keep it as one word.",
    "2. What is the average rating value for the movie Hollow Man ?",
    "3. What is the oldest movie ?",
    "4. Which movies are not reviewed by Darlene Garcia ?",
    "5. What are the co-actors of Kim Dickens, grouped by movie ?",
    "6. List the directors who directed only once.",
    "7. List the British directors and actors",
    "8. List the directors younger than 60 years old",
    "9. List the alive directors",
    "10. Query the database to list comedy movies",
    "11. Which movie genre gets the least ratings by the users ?",
    "12. Which users never co-reviewed together ?",
    "13. What is the proportion of movies reviewed in common by Darlene Garcia and Elizabeth Gonzalez, i.e. review similarity",
    "14. What is the most similar movie to the movie Stanley & Iris in terms of movie genre similarity ?",
    "15. Which movie do you recommend to user Darlene Garcia based on the existing good ratings ? You need to prioritize the movie which gets the most reviews with rating>3.0"
]

request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = [requests.post(CHATBOT_URL, json=data) for data in request_bodies]
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
for i in range(len(questions)):
    print(f"Question {i}: {questions[i]}")
    print(f"Response {i}: {json.loads(outputs[i].content)['output']}")
