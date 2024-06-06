import pickle, sys, time, string
from itertools import combinations
from . import dawg
from concurrent.futures import ThreadPoolExecutor

class Board:
    #constants
    VALUES = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 
        'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 
        'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10, '?': 0
    }
    VALUES.update({char.lower(): 0 for char in VALUES.keys()})

    LETTER_MULTIPLIERS = {(row, col): 1 for row in range(15) for col in range(15)}
    LETTER_MULTIPLIERS.update({
        (0, 3): 2, (1, 5): 3, (2, 6): 2, (2, 8): 2, (3, 7): 2, (1, 9): 3, 
        (5, 5): 3, (9, 9): 3, (5, 9): 3, (0, 11): 2, (3, 14): 2, (5, 13): 3, 
        (6, 12): 2, (7, 11): 2, (8, 12): 2, (9, 13): 3, (11, 14): 2,
        (6, 6): 2, (6, 8):2, (8, 8): 2
    })
    for (row, col), multiplier in list(LETTER_MULTIPLIERS.items()):
        if multiplier > 1:
            LETTER_MULTIPLIERS[(col, row)] = multiplier

    WORD_MULTIPLIERS = {(row, col): 1 for row in range(15) for col in range(15)}
    WORD_MULTIPLIERS.update({
        (1, 1): 2, (2, 2): 2, (3, 3): 2, (4, 4): 2, (7, 7): 2, (13, 13): 2, (10, 4): 2,
        (12, 12): 2, (11, 11): 2, (10, 10): 2, (0, 0): 3, (0, 7): 3, (0, 14): 3, 
        (7, 0): 3, (7, 14): 3, (14, 0): 3, (14, 7): 3, (14, 14): 3, (1, 13): 2, 
        (2, 12): 2, (3, 11): 2, (4, 10): 2, (13, 1): 2, (12, 2): 2, (11, 3): 2
    })

    def __init__(self, dawg, size=15):
        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.dictionary = dawg
        self.is_empty = True

    def place_tile(self, tile, row, col):
        if 0 <= row < self.size and 0 <= col < self.size:
            self.board[row][col] = tile
            self.is_empty = False
        else:
            raise ValueError("Invalid board position")
        
    def clear(self):
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)] 
        self.is_empty = True

    def transpose_board(self):
        self.board = [list(row) for row in zip(*self.board)]

    def display_letter_multipliers(self):
        for row in range(self.size):
            print("|", end="")
            for col in range(self.size):
                multiplier = self.WORD_MULTIPLIERS[(row, col)]
                print(f" {multiplier} ", end="|")
            print("\n" + "-" * (4 * self.size + 1))

    def display_board(self):
        print("Oh shit! Here's the current board!")
        for row in self.board:
            print("|", end="")
            for square in row:
                if square:
                    print(f" {square} ", end="|")
                else:
                    print("   ", end="|")
            print("\n" + "-" * (4 * len(row) + 1))

    def cross_check_sets(self, row):
        valid_letters = []
        ceiling_row = self.board[row - 1] if row > 0 else [None for _ in range(15)]
        floor_row = self.board[row + 1] if row < 14 else [None for _ in range(15)]
        for i in range(15):
            if self.board[row][i] is not None:
                valid_letters.append(set())
            else:
                valid_letters.append(self.cross_check_set(ceiling_row, floor_row, row, i))

        return valid_letters
    
    def cross_check_set(self, ceiling_row, floor_row, row, col):
        valid_letters = set(string.ascii_uppercase)
        top_word = ""
        bottom_word = ""
        i = row
        if ceiling_row[col] is not None:
            while i > 0 and self.board[i - 1][col] is not None:
                top_word = self.board[i - 1][col] + top_word
                i -= 1
        if floor_row[col] is not None:
            while row < 14 and self.board[row + 1][col] is not None:
                bottom_word += self.board[row + 1][col]
                row += 1

        if top_word or bottom_word:
            for letter in string.ascii_uppercase:
                if self.dictionary.lookup(top_word + letter + bottom_word) is None:
                    valid_letters.remove(letter)

        return valid_letters

    def playable_coordinates(self, word, row_index, cross_check_sets, rack):
        lst = self.board[row_index]
        valid_col_idxs = []
        
        for i in range(15 - len(word) + 1):
            if (i > 0 and lst[i-1] is not None) or (i+len(word) < 15 and lst[i+len(word)] is not None):
                continue
            is_anchored = False
            rack_list = list(rack)
            for j, letter in enumerate(word):
                if lst[i+j] is None:
                    if not self.exists_in_rack(letter, rack_list): 
                        break
                    if letter.upper() not in cross_check_sets[i+j]:
                        break
                    if row_index > 0 and self.board[row_index-1][i+j] is not None:
                        is_anchored = True
                    if row_index < 14 and self.board[row_index+1][i+j] is not None:
                        is_anchored = True
                else:
                    if lst[i+j] != letter:
                        break
                    is_anchored = True
                if j == len(word) - 1 and (is_anchored or (self.is_empty and self.touches_star(i, j + 1))):
                    valid_col_idxs.append(i)
                
        return valid_col_idxs

    def all_words_for_row(self, row_index, rack):
        playable_candidates = []
        row_segs = self.extract_row_members(self.board[row_index])
        row_candidates = set(self.dictionary.words_containing_segs(rack, row_segs))
        
        for row_seg in row_segs:
            row_candidates.discard(row_seg)
        cross_check_sets = self.cross_check_sets(row_index)
        for candidate in row_candidates:
            playable_coords = self.playable_coordinates(candidate, row_index, cross_check_sets, rack)
            if playable_coords:
                playable_candidates.append((candidate, playable_coords))
        
        return playable_candidates
    
    def all_valid_plays(self, rack):
        valid_plays = {}
        for row in range(15):
            valid_plays[row] = [] if self.is_empty else self.all_words_for_row(row, rack)
        if self.is_empty:
            valid_plays[7] = self.all_words_for_row(7, rack)

        self.transpose_board()
        for col in range(15):
            valid_plays[col + 15] = [] if self.is_empty else self.all_words_for_row(col, rack) 
        if self.is_empty:
            valid_plays[22] = self.all_words_for_row(7, rack)
        
        return valid_plays

    def sort_plays(self, plays):
        complete_list = []
        # assuming the board is currently in transposed form
        for i in range(29, -1, -1):
            row = i if i < 15 else i - 15
            for tup in plays[i]:
                for col in tup[1]:
                    coord = (row, col, 0) if i < 15 else (col, row, 1)
                    word = tup[0]
                    complete_list.append((word, coord, self.calculate_score(word, row, col)))
            if i == 15:
                self.transpose_board() 

        return sorted(complete_list, key=lambda x: x[2], reverse=True)
    
    def propagation_add(self, row, col):
        row_copy = row
        side_score = 0
        while row_copy > 0 and self.board[row_copy - 1][col] is not None:
            side_score += self.VALUES[self.board[row_copy - 1][col]]
            row_copy -= 1
        while row < 14 and self.board[row + 1][col] is not None:
            side_score += self.VALUES[self.board[row + 1][col]]
            row += 1

        return side_score

    def calculate_side_scores_sum(self, side_branch, word, row, col):
        score_sum = 0
        side_score = 0
        word_mult = 1
        for j, item in enumerate(side_branch):
            if item:
                side_score += self.propagation_add(row, col + j)

                blank = self.board[row][col + j] is None
                letter_mult = self.LETTER_MULTIPLIERS[(row, col + j)] if blank else 1
                side_score += self.VALUES[word[j]] * letter_mult

                word_mult = max(self.WORD_MULTIPLIERS[(row, col + j)] if blank else 1, word_mult)
                side_score *= word_mult
                score_sum += side_score

            word_mult = 1
            side_score = 0

        return score_sum

    def calculate_score(self, word, row, col):
        score = 0
        word_mult = 1
        side_branch = []
        placed_tiles = 0

        for i, letter in enumerate(word):
            blank = self.board[row][col + i] is None
            placed_tiles = placed_tiles + 1 if blank else placed_tiles

            letter_mult = self.LETTER_MULTIPLIERS[(row, col + i)] if blank else 1
            score += self.VALUES[letter] * letter_mult

            word_mult *= self.WORD_MULTIPLIERS[(row, col + i)] if blank else 1
            letter_top = row > 0 and self.board[row - 1][col + i] is not None
            letter_bottom = row < 14 and self.board[row + 1][col + i] is not None
            side_branch.append(blank and (letter_top or letter_bottom))
        
        score *= word_mult
        bingo_bonus = 50 if placed_tiles == 7 else 0

        return score + bingo_bonus + self.calculate_side_scores_sum(side_branch, word, row, col)

    @staticmethod
    def touches_star(start_idx, word_len):
        if start_idx <= 7:
            return start_idx + word_len > 7
        else:
            return False

    @staticmethod
    def exists_in_rack(letter, rack):
        if letter in rack:
            rack.remove(letter)
            return True
        elif '?' in rack and letter.islower():
            rack.remove('?')
            return True
        
        return False

    @staticmethod
    def extract_row_members(lst):
        row_strs = []
        curr_chars = ""
        for id in range(15):
            if lst[id] is not None:
                curr_chars += lst[id]   
            elif curr_chars != "": 
                row_strs.append(curr_chars)
                curr_chars = ""
        if curr_chars != "":
            row_strs.append(curr_chars) 

        return row_strs

    