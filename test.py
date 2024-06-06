import pickle, time
from solver.dawg import Dawg, DawgNode
from solver.board import Board

if __name__ == "__main__":   
    with open("solver/csw.pickle", "rb") as input_file:
        # Load the object from the file using pickle.load
        dawg = pickle.load(input_file)
    #QUERY = sys.argv[1]
    rack = "JFOED??"
    start_time = time.time()

    board = Board(dawg)
    board.place_tile('B', 0, 2)
    board.place_tile('R', 0, 3)
    board.place_tile('E', 0, 4)
    board.place_tile('E', 0, 5)
    board.place_tile('D', 0, 6)
    board.place_tile('G', 0, 9)
    board.place_tile('I', 0, 10)
    board.place_tile('o', 2, 2)
    board.place_tile('O', 2, 3)
    board.place_tile('Z', 2, 4)
    board.place_tile('E', 2, 5)
    board.place_tile('E', 1, 6)
    board.place_tile('L', 1, 7)
    board.place_tile('A', 1, 8)
    board.place_tile('I', 1, 9)
    board.place_tile('D', 1, 10)
    board.place_tile('R', 1, 5)
    board.place_tile('M', 2, 9)
    board.place_tile('P', 3, 9)
    board.place_tile('E', 3, 10)
    board.place_tile('R', 3, 11)
    board.place_tile('S', 3, 12)
    board.place_tile('O', 3, 13)
    board.place_tile('N', 3, 14)
    board.place_tile('O', 2, 14)
    board.place_tile('O', 1, 14)
    board.place_tile('G', 0, 14)
    board.place_tile('I', 4, 9)
    board.place_tile('N', 5, 9)
    board.place_tile('G', 6, 9)
    board.place_tile('G', 6, 10)
    board.place_tile('A', 6, 8)
    board.place_tile('M', 6, 7)
    board.place_tile('E', 7, 7)
    board.place_tile('S', 8, 7)
    board.place_tile('S', 9, 7)
    board.place_tile('A', 10, 7)
    board.place_tile('G', 11, 7)
    board.place_tile('I', 12, 7)
    board.place_tile('N', 13, 7)
    board.place_tile('G', 14, 7)

    # if (dawg.lookup(QUERY)):
    #     print(f"{QUERY} exists in dictionary")
    # else:
    #     print(f"{QUERY} does not exist in dictionary")
    board.display_board()
    # board.transpose_board()
    # shit = board.all_words_for_row(5, rack)
    # print(shit)

    valid_words = board.all_valid_plays(rack)
    print(f"Time taken: {time.time() - start_time}")
    plays = board.sort_plays(valid_words)
    print(f"Time taken: {time.time() - start_time}")
    print(plays[:100])

    #print(valid_words[int(QUERY)])
    # print(board.calculate_score("XELAID", 1, 5))
    # breeds = dawg.words_containing_string(row_chars)
    # expanded_breds = expand_list(breeds, row_chars, 1)
    # row_candidates = dawg.words_containing_chars(rack) + expanded_breds
    # superimposable_candidates = []

    # for candidate in row_candidates:
    #     if is_superimposable(candidate, board.board[0], rack_list):
    #         superimposable_candidates.append(candidate)
    
    #print(is_superimposable("bOTA", board.board[0], rack_list))
    # print(superimposable_candidates)
    # print(len(superimposable_candidates))