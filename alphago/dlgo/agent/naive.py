import random
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.goboard import Move
from dlgo.gotypes import Point

class RandomBot(Agent):
    def select_move(self, game_state):
        # 合法手が打てる交点を全て列挙する。
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r,col=c)
                if game_state.is_valid_move(Move.play(candidate)) and \
                    not is_point_an_eye(game_state.board,
                    candidate,
                    game_state.next_player
                    ):
                    candidates.append(candidate)
            if not candidates:
                return Move.pass_turn()
        # 列記した交点の中から１交点を選んでプレイする。
        return Move.play(random.choice(candidates))

def best_result(game_state):
    # ゲームが終わっているなら結果を返す。
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return GameResult.win
        elif game_state.winner() is None:
            return GameResult.draw
        else:
            return GameResult.loss
    # 候補手の探索
    best_result_so_far = GameResult.loss
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)     # <1>
        opponent_best_result = best_result(next_state)         # <2>
        our_result = reverse_game_result(opponent_best_result) # <3>
        if our_result.value > best_result_so_far.value:        # <4>
            best_result_so_far = our_result
    return best_result_so_far

class MinimaxAgent(Agent):
    def select_move(self, game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []
        for possible_move in game_state.legal_moves():
            next_state = game_state.apply_move(possible_move)
            opponent_best_outcome = best_result(next_state)
            our_best_outcome = reverse_game_result(opponent_best_outcome)
            if our_best_outcome == GameResult.win:
                winning_moves.append(possible_move)
            elif our_best_outcome == GameResult.draw:
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)
        if winning_moves:
            return random.choice(winning_moves)
        if draw_moves:
            return random.choice(draw_moves)
        return random.choice(losing_moves)