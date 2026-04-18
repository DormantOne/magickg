"""
Go Game Engine — Small board (default 7x7)
Full rules: placement, capture, ko, pass, territory scoring.
"""
BLACK = 1
WHITE = 2
EMPTY = 0

def opponent(color):
    return WHITE if color == BLACK else BLACK

class GoGame:
    def __init__(self, size=7, komi=3.5, move_cap=80):
        self.size = size
        self.komi = komi
        self.move_cap = move_cap
        self.board = [[EMPTY]*size for _ in range(size)]
        self.current = BLACK
        self.history = []
        self.captures = {BLACK: 0, WHITE: 0}
        self.prev_board = None
        self.consecutive_passes = 0
        self.game_over = False
        self.winner = None
        self.move_count = 0

    def copy_board(self):
        return [row[:] for row in self.board]

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def group_and_liberties(self, r, c):
        color = self.board[r][c]
        if color == EMPTY:
            return set(), set()
        visited = set()
        liberties = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in self.neighbors(cr, cc):
                if self.board[nr][nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr][nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return visited, liberties

    def remove_group(self, group):
        for r, c in group:
            self.board[r][c] = EMPTY

    def play(self, color, row, col):
        if row == -1 and col == -1:
            self.consecutive_passes += 1
            self.history.append((color, "pass"))
            self.current = opponent(color)
            self.move_count += 1
            if self.consecutive_passes >= 2:
                self.end_game()
            return True, "pass"
        self.consecutive_passes = 0
        if not self.in_bounds(row, col):
            return False, "out of bounds"
        if self.board[row][col] != EMPTY:
            return False, "occupied"
        old_board = self.copy_board()
        self.board[row][col] = color
        captured = 0
        for nr, nc in self.neighbors(row, col):
            if self.board[nr][nc] == opponent(color):
                group, libs = self.group_and_liberties(nr, nc)
                if len(libs) == 0:
                    self.remove_group(group)
                    captured += len(group)
        own_group, own_libs = self.group_and_liberties(row, col)
        if len(own_libs) == 0:
            self.board = old_board
            return False, "suicide"
        if self.prev_board and self.board == self.prev_board:
            self.board = old_board
            return False, "ko"
        self.prev_board = old_board
        self.captures[color] += captured
        self.history.append((color, row, col))
        self.current = opponent(color)
        self.move_count += 1
        if self.move_cap and self.move_count >= self.move_cap:
            self.end_game()
        return True, f"({row},{col})" + (f" cap:{captured}" if captured else "")

    def end_game(self):
        self.game_over = True
        b, w = self.score()
        if b > w: self.winner = BLACK
        elif w > b: self.winner = WHITE
        else: self.winner = None

    def score(self):
        territory = {BLACK: 0, WHITE: 0}
        visited = [[False]*self.size for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                if visited[r][c] or self.board[r][c] != EMPTY:
                    continue
                region = set()
                borders = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in region: continue
                    region.add((cr, cc))
                    visited[cr][cc] = True
                    for nr, nc in self.neighbors(cr, cc):
                        if self.board[nr][nc] == EMPTY and (nr, nc) not in region:
                            stack.append((nr, nc))
                        elif self.board[nr][nc] != EMPTY:
                            borders.add(self.board[nr][nc])
                if len(borders) == 1:
                    territory[borders.pop()] += len(region)
        stones = {BLACK: 0, WHITE: 0}
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] != EMPTY:
                    stones[self.board[r][c]] += 1
        b = stones[BLACK] + territory[BLACK] + self.captures[BLACK]
        w = stones[WHITE] + territory[WHITE] + self.captures[WHITE] + self.komi
        return b, w

    def legal_moves(self, color):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] != EMPTY: continue
                old = self.copy_board()
                old_prev = self.prev_board
                self.board[r][c] = color
                valid = True
                for nr, nc in self.neighbors(r, c):
                    if self.board[nr][nc] == opponent(color):
                        g, l = self.group_and_liberties(nr, nc)
                        if len(l) == 0:
                            for gr, gc in g: self.board[gr][gc] = EMPTY
                _, own_libs = self.group_and_liberties(r, c)
                if len(own_libs) == 0: valid = False
                if valid and self.prev_board and self.board == self.prev_board: valid = False
                self.board = old
                self.prev_board = old_prev
                if valid: moves.append((r, c))
        moves.append((-1, -1))
        return moves

    def board_string(self):
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        cols = "  " + " ".join(chr(65+i) for i in range(self.size))
        lines = [cols]
        for r in range(self.size):
            row_str = f"{r} " + " ".join(sym[self.board[r][c]] for c in range(self.size)) + f" {r}"
            lines.append(row_str)
        lines.append(cols)
        return "\n".join(lines)

    def board_array(self):
        return [self.board[r][c] for r in range(self.size) for c in range(self.size)]

    def state_dict(self):
        b, w = self.score()
        return {"size": self.size, "board": self.board_array(),
                "current": self.current, "current_name": "black" if self.current == BLACK else "white",
                "captures": self.captures, "move_count": self.move_count,
                "game_over": self.game_over, "winner": self.winner,
                "winner_name": {BLACK:"black",WHITE:"white",None:"draw"}.get(self.winner,"?"),
                "history": self.history[-20:],
                "score": {"black": b, "white": w},
                "consecutive_passes": self.consecutive_passes}
