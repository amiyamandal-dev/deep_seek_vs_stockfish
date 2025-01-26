import random
from ollama import generate
import chess
import chess.engine
import sys
import re
from loguru import logger

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
            return True, "ILLEGAL"
        self.board.push(move)
        if self.board.is_game_over():
            return True, self._get_game_result()
        result = self.engine.play(self.board, limit=chess.engine.Limit(time=self.time_limit))
        self.board.push(result.move)
        if self.board.is_game_over():
            return True, self._get_game_result()
        return False, "ONGOING"

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
        moves = [str(move) for move in self.board.legal_moves]
        return moves

    def close(self):
        if self.engine:
            self.engine.quit()


class LLMChessAgent:
    def __init__(self, env: ChessEnv, model="deepseek-r1:14b"):
        self.model = model
        self.env = env
        self.pattern = re.compile(r"\s*([a-h][1-8][a-h][1-8][qrbn]?)")

    def get_action(self, opponent_move_uci, history=[]):
        if opponent_move_uci == "None":
            return "e2e4"
        moves = self.env.get_all_valid_moves()
        prompt = f"""As a chess expert, your goal is to demonstrate your expertise by winning a game of chess. 
Your task is to strategize and play a game of chess with the intent of achieving victory. 
You are White.
NOTE:- generate only 'ONE' valid move.

ALL Valid Moves: {",".join(moves)}
HISTORY:
{"\n".join(history)}

Output example:
<UCI>=e2e4
<UCI>=h2h3
<UCI>=<best move what you have thought from valid moves>
"""
        logger.info(f"prompt: {prompt}")
        response = generate(model=self.model, prompt=prompt)
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
            random_item = random.choice(moves)
            return random_item


def play_game_llm_vs_stockfish():
    STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'
    env = ChessEnv(stockfish_path=STOCKFISH_PATH, time_limit=0.1)
    agent = LLMChessAgent(env=env, model="deepseek-r1:14b")
    board = env.reset()
    moves_history = []
    done = False
    status = "ONGOING"
    last_opponent_move = "None"
    while not done:
        white_move = agent.get_action(opponent_move_uci=last_opponent_move, history=moves_history)
        logger.info(f"white_move: {white_move}")
        moves_history.append(f"You: {white_move}")
        done, status = env.step(white_move)
        if done:
            break
        black_move = board.move_stack[-1].uci()
        logger.info(f"black_move: {black_move}")
        moves_history.append(f"Opponent: {black_move}")
        last_opponent_move = black_move
        print("\nFinal Board Position:")
        print(env.board)
        print("\nGame Status:", status)
        print("\nMove History:")
    env.close()

if __name__ == "__main__":
    play_game_llm_vs_stockfish()
