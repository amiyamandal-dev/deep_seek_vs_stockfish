import random
import chess
import chess.engine
import os
import sys
import re
from loguru import logger
from ollama import generate

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv("API_KEY"), 
    base_url="https://api.deepseek.com"
)


class ChessEnv:
    def __init__(self, stockfish_path, time_limit=0.5):
        self.stockfish_path = stockfish_path
        self.time_limit = time_limit
        self.engine = None
        self.board = None

    def reset(self):
        if self.engine is not None:
            self.engine.quit()
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except FileNotFoundError:
            print("Stockfish executable not found. Update STOCKFISH_PATH.")
            sys.exit(1)
        self.board = chess.Board()
        return self.board

    def step(self, white_move_uci):
        move = chess.Move.from_uci(white_move_uci)
        if move not in self.board.legal_moves:
            return True, "ILLEGAL", None

        self.board.push(move)
        analysis = self.engine.analyse(self.board, limit=chess.engine.Limit(depth=10))
        score_obj = analysis["score"].pov(chess.WHITE)
        if score_obj.is_mate():
            white_score = f"Mate in {score_obj.mate()}"
        else:
            cp = score_obj.score(mate_score=100000)
            white_score = str(cp)
        logger.info(f"White's move: {white_move_uci} => Score: {white_score}")

        if self.board.is_game_over():
            return True, self._get_game_result(), white_score

        result = self.engine.play(self.board, limit=chess.engine.Limit(time=self.time_limit))
        self.board.push(result.move)

        if self.board.is_game_over():
            return True, self._get_game_result(), white_score

        return False, "ONGOING", white_score

    def _get_game_result(self):
        if self.board.is_checkmate():
            return "BLACK_WINS" if self.board.turn else "WHITE_WINS"
        if self.board.is_stalemate():
            return "DRAW"
        if self.board.is_insufficient_material():
            return "DRAW"
        if self.board.can_claim_threefold_repetition():
            return "DRAW"
        if self.board.can_claim_fifty_moves():
            return "DRAW"
        return "UNKNOWN_END"
    
    def get_all_valid_moves(self):
        return [str(move) for move in self.board.legal_moves]

    def close(self):
        if self.engine:
            self.engine.quit()


class LLMChessAgent:
    def __init__(self, env: ChessEnv, model="deepseek-r1:14b", use_ollama=True):
        self.model = model
        self.env = env
        self.use_ollama = use_ollama
        self.pattern = re.compile(r"\s*([a-h][1-8][a-h][1-8][qrbn]?)")

    def get_action(self, opponent_move_uci, history=[]):
        if opponent_move_uci == "None":
            return "e2e4"
        moves = self.env.get_all_valid_moves()

        # Build a single prompt (the "user" message).
        user_prompt = f"""
        As a chess expert, your goal is to demonstrate your expertise by winning a game of chess. Your task is to strategize and play a game of chess with the intent of achieving victory. You will analyze your opponent's moves and your move to generate the best possible 'ONE' move. You are White.

<think>
Step 1: Assess the current board position, considering material balance, piece activity, All Valid moves, and control of key squares.
Step 2: Identify potential threats from the opponent and opportunities for you.
Step 3: Evaluate candidate moves, considering both immediate tactics and long-term strategy.
Step 4: Select the move that maximizes your advantage while minimizing potential risks.
</think>

NOTE: Generate only 'ONE' valid move in UCI format (e.g., e2e4).


ALL Valid Moves: {",".join(moves)}
HISTORY:
{"\n".join(history)}

Output format example:
<UCI>=e2e4
<UCI>=h2h3
<UCI>=<best move from valid moves>
"""

        # Use DeepSeek's Chat API:
        logger.info(f"prompt: {user_prompt}")
        if self.use_ollama:
            response = generate(model=self.model, prompt=user_prompt)
            agent_move = response.response.strip()
            agent_move = agent_move.split("</think>")[-1].strip()
            logger.info(f"response: {agent_move}")
            uci_move_2 = agent_move.replace("<UCI>=", "").strip()
            logger.info(f"Extracted move:{uci_move_2}")
            match = self.pattern.search(uci_move_2)
            try:
                if match:
                    print(match)
                    uci_move_2 = match.group(1)
                    return uci_move_2.strip().replace("\n", "")
            except:
                logger.error("Error in extracting move")
                return random.choice(moves) 
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an advanced chess model."},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False
            )
            agent_move = response.choices[0].message.content.strip()
            logger.info(f"response: {agent_move}")

            agent_move = agent_move.split("</think>")[-1].strip()
            uci_move_2 = agent_move.replace("<UCI>=", "").strip()
            logger.info(f"Extracted move: {uci_move_2}")

            match = self.pattern.search(uci_move_2)
            try:
                if match:
                    print(match)
                    uci_move_2 = match.group(1)
                    return uci_move_2.strip().replace("\n", "")
            except:
                logger.error("Error in extracting move")
                return random.choice(moves)


def play_game_llm_vs_stockfish():
    STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'
    env = ChessEnv(stockfish_path=STOCKFISH_PATH, time_limit=0.1)
    agent = LLMChessAgent(env=env, model="deepseek-reasoner", use_ollama=False)
    board = env.reset()
    moves_history = []
    done = False
    status = "ONGOING"
    last_opponent_move = "None"

    while not done:
        white_move = agent.get_action(
            opponent_move_uci=last_opponent_move,
            history=moves_history
        )
        logger.info(f"white_move: {white_move}")
        done, status, white_score = env.step(white_move)

        moves_history.append(f"You: {white_move} (score={white_score})")
        if done:
            break

        black_move = board.move_stack[-1].uci()
        logger.info(f"black_move: {black_move}")
        moves_history.append(f"Opponent: {black_move}")

        last_opponent_move = black_move
        logger.info("Final Board Position:")
        logger.info(f"\n\n{env.board}")
        logger.info(f"Game Status:{status}", )

    env.close()

if __name__ == "__main__":
    play_game_llm_vs_stockfish()
