from ollama import chat, generate
from ollama import ChatResponse


response = generate(model='deepseek-r1:8b', prompt='''As a chess expert, your goal is to demonstrate your expertise by winning a game of chess. Your task is to strategize and play a game of chess with the intent of achieving victory. You will analyze your opponent\'s moves and your Input and generate best possible 'ONE' move.
                Opponent Move: e2e4\n\n
                My Move:''')

print(response.response)