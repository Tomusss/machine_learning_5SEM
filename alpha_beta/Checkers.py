import copy 
import time
ROWS = 8
COLS = 8
WHITE = 1
WHITE_Q = 2
BLACK = -1
BLACK_Q = -2 
EMPTY = 0 


class Board():
    def __init__(self):
        self.board = []
        self.white = 12 
        self.black = 12

        self.create()
    def create(self):
        self.board=[]
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if (row + col) % 2 == 1:
                    if row < 3:
                        self.board[row].append(BLACK)
                    elif row > 4:
                        self.board[row].append(WHITE)
                    else: 
                        self.board[row].append(EMPTY)
                else:
                    self.board[row].append(EMPTY)

    def print(self):
        print("   0 1 2 3 4 5 6 7") 
        print(" +-----------------+")
        for row in range(ROWS):
            print(f'{row} |', end = '')
            for col in range(COLS):
                el = self.board[row][col]
                if el == EMPTY:
                    if (row+col)%2 == 1: print('.', end = ' ')
                    else: print(' ', end= ' ')
                elif el == WHITE: print('o', end=' ')
                elif el == WHITE_Q: print('O', end=' ')
                elif el == BLACK: print('x', end=' ')
                elif el == BLACK_Q: print('X', end=' ')
            print('')

    def evaluate(self):
        heatmap = [
            [0, 4, 0, 4, 0, 4, 0, 4],
            [4, 0, 3, 0, 3, 0, 3, 0],
            [0, 3, 0, 4, 0, 4, 0, 4],
            [4, 0, 5, 0, 5, 0, 3, 0],
            [0, 3, 0, 5, 0, 5, 0, 4],
            [4, 0, 4, 0, 3, 0, 3, 0],
            [0, 4, 0, 4, 0, 4, 0, 4], 
            [4, 0, 4, 0, 4, 0, 4, 0], 
        ]

        white_score = 0
        black_score = 0

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                
                if piece == EMPTY:
                    continue
                
                pos_bonus = heatmap[row][col]

                if piece == WHITE:
                    white_score += 10 + pos_bonus
                elif piece == WHITE_Q:
                    white_score += 20 + pos_bonus
                elif piece == BLACK:
                    black_score += 10 + pos_bonus
                elif piece == BLACK_Q:
                    black_score += 20 + pos_bonus

        return white_score - black_score

    def count(self, el):
        count = 0
        for row in self.board:
            for each in row: 
                if each == el:
                    count += 1 
        return count 

    def all_moves(self, color):
        moves = []
        
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                
                if piece == color: # wersja dla pionka 
                    move_dirs = [-1] if color == WHITE else [1]
                    kill_dirs = [-1, 1] 
                    
                    # zwykly ruch
                    for r_step in move_dirs:
                        for c_step in [-1, 1]:
                            if 0 <= row+r_step < ROWS and 0 <= col+c_step < COLS:
                                if self.board[row+r_step][col+c_step] == EMPTY:
                                    new_b = copy.deepcopy(self)
                                    new_b.move(color, row, col, row+r_step, col+c_step)
                                    moves.append(new_b)
                    
                    # bicie
                    for r_step in kill_dirs:
                        for c_step in [-1, 1]:
                             if 0 <= row+(r_step*2) < ROWS and 0 <= col+(c_step*2) < COLS:
                                 neighbor = self.board[row+r_step][col+c_step]
                                 if neighbor != EMPTY and neighbor * color < 0:
                                     if self.board[row+r_step*2][col+c_step*2] == EMPTY:
                                         new_b = copy.deepcopy(self)
                                         new_b.move(color, row, col, row+r_step*2, col+c_step*2)
                                         moves.append(new_b)

                elif piece == (color * 2): # damka na przekatnej 
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    
                    for r_dir, c_dir in directions:
                        e_found = False
                        
                        # sprawdzamy do potencjalnego konca plansszy 
                        for i in range(1, 8):
                            tgt_r = row + r_dir*i
                            tgt_c = col + c_dir*i
                            
                            # koniec planszyy
                            if not (0 <= tgt_r < ROWS and 0 <= tgt_c < COLS):
                                break
                                
                            target = self.board[tgt_r][tgt_c]
                            
                            if target == EMPTY:
                                if not e_found: # zwykly ruch
                                    new_b = copy.deepcopy(self)
                                    new_b.move(color, row, col, tgt_r, tgt_c)
                                    moves.append(new_b)
                                else:
                                    new_b = copy.deepcopy(self) # po bicu
                                    new_b.move(color, row, col, tgt_r, tgt_c)
                                    moves.append(new_b)
                                    
                            elif target * color > 0:
                                break # swoj pionek 
                                
                            elif target * color < 0:
                                #wrog
                                if e_found:
                                    break # drugi wrog w tej samej lini
                                e_found = True # zaznaczamy wroga
                            
        return moves
    
    def move(self, color, old_row, old_col, new_row, new_col):
        piece = self.board[old_row][old_col]
        self.board[old_row][old_col] = EMPTY
        self.board[new_row][new_col] = piece 

        if abs(piece) == 1:
            if color == WHITE and new_row == 0:  # damki 
                self.board[new_row][new_col] = WHITE_Q
            elif color == BLACK and new_row == ROWS - 1:
                self.board[new_row][new_col] = BLACK_Q

        if abs(new_row - old_row) > 1:
            r_step = 1 if new_row > old_row else -1
            c_step = 1 if new_col > old_col else -1
            
            current_r, current_c = old_row + r_step, old_col + c_step
            while current_r != new_row:
                if self.board[current_r][current_c] != EMPTY:
                    if self.board[current_r][current_c] > 0: 
                        self.white -= 1
                    else:
                        self.black -= 1
                    self.board[current_r][current_c] = EMPTY
                    break 
                current_r += r_step
                current_c += c_step
                
def get_human_move(board, color):
    legal = board.all_moves(color)
    
    if not legal:
        return None, None 
    
    print(f"\n--- RUCH {'BIAŁYCH (o/O)' if color == WHITE else 'CZARNYCH (x/X)'} ---")
    while True:
        try:
            which = input("Podaj rząd i kolumnę PIONKA (np. 5 2): ")
            target = input("Podaj rząd i kolumnę CELU   (np. 4 3): ")
            
            r1, c1 = map(int, which.split())
            r2, c2 = map(int, target.split())
                        
            temp_board = copy.deepcopy(board)
            # sprawdzenie czy nasz pionek 
            piece = board.board[r1][c1]
            if piece == EMPTY or (color == WHITE and piece < 0) or (color == BLACK and piece > 0):
                print("Błąd przy wyborze pionka")
                continue

            temp_board.move(color, r1, c1, r2, c2)
            
            # sprawdzenie czy ruch jest legalny
            found_move = None
            for leg_move in legal:
                if leg_move.board == temp_board.board:
                    found_move = leg_move
                    break
            
            if found_move:
                return found_move.evaluate(), found_move
            else:
                print("Błąd przy wyborze pola")
                
        except ValueError:
            print("Błąd formatu")
        except IndexError:
            print("Współrzędne poza planszą")
        except Exception as e:
            print(f"Nieoczekiwany błąd: {e}")

def alphabeta(board, dep, alpha, beta, max_player):
    if dep == 0 or board.black == 0 or board.white  == 0:
        return board.evaluate(), board 
    if max_player:
        max_eval = float('-inf')
        best_m = None
        for move in board.all_moves(WHITE):
            ev, _ = alphabeta(move,dep-1,alpha,beta,False)
            if ev > max_eval:
                max_eval = ev
                best_m = move
            alpha = max(max_eval,alpha)
            if beta <= alpha:
                break
        return max_eval, best_m
    
    else:
        min_eval = float("inf")
        best_m = None
        for move in board.all_moves(BLACK):
            ev, _ = alphabeta(move,dep-1,alpha,beta,True)
            if ev < min_eval:
                min_eval = ev
                best_m = move
            beta = min(ev,beta)
            if beta <= alpha:
                break 
        return min_eval, best_m


def main():
    game = Board()
    game.create()

    depth = 5
    turn = WHITE 
    
    print("-----------------------")
    print("Grasz białympi pionkami, komputer czarnymi")
    print("Zaczynają białe pionki")
    PLAYER = WHITE
    AI = BLACK
    turn = WHITE

    while True:
        game.print()
        
        if game.white == 0:
            print("Wygrały x")
            break
        if game.black == 0:
            print("Wygrały O")
            break

        if turn == PLAYER:
            print("Ruch bialych")
            value, new_board = get_human_move(game,PLAYER)
        else:
            print("Ruch czarnych")
            is_maximizing = True if AI == WHITE else False
            value, new_board = alphabeta(game, depth, float('-inf'), float('inf'), is_maximizing)

        if new_board is None:
            winner = "CZARNE" if turn == WHITE else "BIAŁE"
            print(f"Wygrywają {winner}.")
            break

        game = new_board
        print(f"evaluation: {value}")
        
        turn = BLACK if turn == WHITE else WHITE


if __name__ == '__main__':
    main()