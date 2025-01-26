import random
from ollama import generate
import chess
import chess.engine
import sys
import re
from loguru import logger

class ChessEnv:
    
    def __init__(self, stockfish_path, time_limit=0.5):
        """
        :param stockfish_path: Path to your Stockfish executable.
        :param time_limit: Time in seconds for Stockfish to think per move.
        """
        self.stockfish_path = stockfish_path
        self.time_limit = time_limit
        self.engine = None
        self.board = None

    def reset(self):
        """Resets the environment with a fresh board and Stockfish engine."""
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
        """
        1. White (the LLM) makes a move.
        2. Check if the game is over.
        3. If not over, let Stockfish make a move.
        4. Check if the game is over again.
        
        :param white_move_uci: The White move in UCI format (e.g. "e2e4").
        :return: done (bool), status (str)
                 'ILLEGAL' if White's move is invalid
                 'WHITE_WINS', 'BLACK_WINS', 'DRAW', or 'ONGOING'
        """
        # Validate White’s move
        move = chess.Move.from_uci(white_move_uci)
        if move not in self.board.legal_moves:
            # Illegal move => immediate termination
            return True, "ILLEGAL"

        # Make White's move
        self.board.push(move)
        # Check if game ended
        if self.board.is_game_over():
            return True, self._get_game_result()

        # Stockfish (Black) move
        result = self.engine.play(self.board, limit=chess.engine.Limit(time=self.time_limit))
        self.board.push(result.move)
        # Check again
        if self.board.is_game_over():
            return True, self._get_game_result()

        return False, "ONGOING"

    def _get_game_result(self):
        """Classify the final board state."""
        if self.board.is_checkmate():
            # If it's White's turn after checkmate => Black delivered mate
            # If it's Black's turn => White delivered mate
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
        """Close the engine process."""
        if self.engine:
            self.engine.quit()


class LLMChessAgent:
    """
    A simple agent that calls an open-source LLM (via Ollama) to get its move.
    The prompt always includes the opponent's last move and asks for White’s move.
    """
    def __init__(self, env:ChessEnv, model="deepseek-r1:14b"):
        self.model = model
        self.env = env
        self.pattern = re.compile(r"\s*([a-h][1-8][a-h][1-8][qrbn]?)")
        

    def get_action(self, opponent_move_uci, history=[]):
        """
        Build a prompt around the opponent’s last move. 
        We expect the LLM to return a single move in UCI format.
        """
        if opponent_move_uci == "None":
            return  "e2e4"
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
        
        # We assume the LLM’s response is a raw string with something like "e7e5".
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
    """
    Demonstrates a single game where:
      - White's moves come from the LLM (via Ollama).
      - Black's moves come from Stockfish.
    We track and display all moves.
    """

    # 3.1 Create environment and agent
    STOCKFISH_PATH ='/opt/homebrew/bin/stockfish'  # Update this!
    env = ChessEnv(stockfish_path=STOCKFISH_PATH, time_limit=0.1)
    agent = LLMChessAgent(env= env, model="deepseek-r1:14b")

    # 3.2 Reset the environment and get initial board
    board = env.reset()

    # We’ll store moves in a list for display
    moves_history = []
    done = False
    status = "ONGOING"

    # 3.3 The game loop
    #   White (LLM) moves first, but the LLM’s prompt needs the last opponent move.
    #   For the very first move, there is no "opponent move," so we’ll pass something like "None" or an empty string.
    last_opponent_move = "None"  

    while not done:
        # Ask the LLM for its move in UCI format
        white_move = agent.get_action(opponent_move_uci=last_opponent_move, history=moves_history)
        logger.info(f"white_move: {white_move}")
        moves_history.append(f"You: {white_move}")

        # Step the environment (this will also cause Stockfish to move)
        done, status = env.step(white_move)
        if done:
            break

        # If the game is still ongoing, the last move in the board state is Stockfish’s move
        black_move = board.move_stack[-1].uci()
        logger.info(f"black_move: {black_move}")
        moves_history.append(f"Opponent: {black_move}")

        # For the next iteration, the LLM sees the last opponent move
        last_opponent_move = black_move

        # 3.4 Print final board and moves
        print("\nFinal Board Position:")
        print(env.board)
        print("\nGame Status:", status)
        print("\nMove History:")

    env.close()

if __name__ == "__main__":
    play_game_llm_vs_stockfish()
