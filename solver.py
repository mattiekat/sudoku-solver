import itertools as itt
import functools as fnt

# This code is largely inspired by https://www.sudokuoftheday.com/techniques/


VALUES = set([x + 1 for x in range(9)])


class InvalidGameState(Exception):
    pass


def row_col_iter():
    """
    Construct an iterator over the rows and columns.
    :return: An iterator over the rows and columns which will return tuples in the form (r, c).
    """
    return map(lambda i: (i // 9, i % 9), range(9**2))


def filter_map(fn, it):
    """
    Map and filter values. If a value is mapped to None, then it will be filtered.
    :param fn: Function used to map values.
    :param it: Values over which the function should run.
    :return: A filter_map iterator which yeilds non-None values mapped by the function.
    """
    return filter(lambda x: x is not None, map(fn, it))


def calc_earmarks(board):
    """
    Calculate a base set of earmarks based on what numbers can obviously not be placed at a given location.

    I.e. The set of possibilities P at a cell must be disjoint from the current values for all other cells in the row R,
    col C, and box B. Thus P = {1,2,3,4,5,6,7,8,9} - (R | C | B).

    :param board: Sudoku board with known values.
    :return: A 9x9 list of sets, each set containing the values which are possible for the each cell at that location..
    """
    def get_row(board, r):
        s = set(board[r])
        s.discard(0)
        return s

    def get_col(board, c):
        s = {board[r][c] for r in range(9)}
        s.discard(0)
        return s

    def get_box(board, r, c):
        s = set()
        for i in range((r // 3) * 3, (r // 3) * 3 + 3):
            for j in range((c // 3) * 3, (c // 3) * 3 + 3):
                s.add(board[i][j])
        s.discard(0)
        return s

    earmarks = [[set() for _ in range(9)] for _ in range(9)]
    for c in range(9):
        col = get_col(board, c)
        box = None
        for r in range(9):
            if r % 3 == 0:
                box = get_box(board, r, c)
            if board[r][c] != 0:
                continue
            row = get_row(board, r)  # row is easiest to calculate, so do it the most
            earmarks[r][c] = VALUES - (row | col | box)

    return earmarks


def update_earmarks(earmarks, newval, r, c):
    """
    Update earmarks if a value is set on the board. This will remove that `newvalue` form the row, col, and box groups
    as it will no longer be possible in any of those locations.
    :param earmarks: 9x9 list of possible values for each cell.
    :param newval: The value which is to be removed removed from the possibilities.
    :param r: Row the new value is placed at.
    :param c: Column the new value is placed at.
    """
    # update row, col, and group to no longer include the newly placed value as an option
    earmarks[r][c] = set()

    # go through column
    for r2 in range(9):
        earmarks[r2][c].discard(newval)

    # go through row
    for c2 in range(9):
        earmarks[r][c2].discard(newval)

    # go through box
    for r2 in range((r // 3) * 3, (r // 3) * 3 + 3):
        for c2 in range((c // 3) * 3, (c // 3) * 3 + 3):
            earmarks[r2][c2].discard(newval)


def set_val(board, earmarks, newval, r, c):
    """
    Set a value on the board and update earmarks appropriately.
    :param board: Sudoku board with known values.
    :param earmarks: 9x9 list of possible values for each cell.
    :param newval: The value which is to be set at the location.
    :param r: Row the new value is placed at.
    :param c: Column the new value is placed at.
    """
    board[r][c] = newval
    update_earmarks(earmarks, newval, r, c)


def simple_fill(board, earmarks):
    """
    Check for single candidate or single position events which mean we can trivially decide what a cell must be.

    Let P be the set of possibilities for a cell.
    A cell is a single candidate if card(P) = 1.

    Let C be a single cell.
    Let P be the possibilities for C.
    Let T be a set of n cells in the same group as C (i.e. row, col, or box and excluding C).
    Let H be the union of all possibilities for cells in T.
    A cell C is a single position if it is a member of a group T such that card(H - P) = 1.

    :param board: Sudoku board with known values.
    :param earmarks: 9x9 list of possible values for each cell.
    :return: Whether any changes were made to board or earmarks.
    """
    def get_possible_row(earmarks, r, c):
        # Values which can be placed somewhere else in the row
        return fnt.reduce(set.__or__, map(lambda c2: set() if c2 == c else earmarks[r][c2], range(9)))

    def get_possible_col(earmarks, r, c):
        # Values which can be placed somewhere else in the column
        return fnt.reduce(set.__or__, map(lambda r2: set() if r2 == r else earmarks[r2][c], range(9)))

    def get_possible_box(earmarks, r, c):
        # Values which can be placed somewhere else in the box
        possible_box = set()
        for r2 in range((r // 3) * 3, (r // 3) * 3 + 3):
            for c2 in range((c // 3) * 3, (c // 3) * 3 + 3):
                if r2 != r or c2 != c:
                    possible_box |= earmarks[r2][c2]
        return possible_box

    changed = set()
    for (r, c) in row_col_iter():
        if board[r][c] != 0:
            continue

        possible = earmarks[r][c]
        if len(possible) <= 0:
            raise InvalidGameState("No possibilities left for ({}, {}).".format(r, c))

        # Single Candidate
        if len(possible) == 1:
            val = possible.pop()
            print("Single Candidate: ({}, {}), Val {}".format(r, c, val))
            set_val(board, earmarks, val, r, c)
            changed.add((r, c))
            continue

        # Single Position
        # check if any of the possible values can only be in this loc for each the row, col, and box.
        possible_row = get_possible_row(earmarks, r, c)
        possible_col = get_possible_col(earmarks, r, c)
        possible_box = get_possible_box(earmarks, r, c)

        for i, possible_x in enumerate([possible_col, possible_row, possible_box]):
            p = possible - possible_x
            if len(p) > 1:
                raise InvalidGameState("Multiple values must be placed at ({}, {}).".format(r, c))
            elif len(p) == 1:
                val = p.pop()
                print("Single Position ({}): ({}, {}), Val {}".format(['Col', 'Row', 'Box'][i], r, c, val))
                set_val(board, earmarks, val, r, c)
                changed.add((r, c))
                break
    return changed


def candidate_lines(earmarks):
    """
    If the possibilities for a given number in a group are all in a line on a row or column, then we can remove those as
    possibilities from the rest of the row/column.

    Let B be a 3x3 box of cells (i.e. one of the sudoku 3x3 boxes).
    Let R be the possibilities for a partial row which is a subset of B (i.e. a 3-long row).
    Let C be the possibilities for a partial column which is a subset of B (i.e. a 3-long column).

    Let R1, R2, R3 in B denote all possibilities of partial rows of B.
    Let C1, C2, C3 in B denote all possibilities of partial columns of B.

    Then, without loss of generality, the set of candidate lines for a given partial row R1 is R1 - (R2 | R3), and
    further, the set of candidate lines for a given partial column C1 is C1 - (C2 | C3).

    :param earmarks: 9x9 list of possible values for each cell.
    :return: Whether any changes to earmarks were made.
    """
    changed = False

    for i in range(9):  # for each of the boxes
        # Calculate the base row and col
        br = (i // 3) * 3
        bc = (i % 3) * 3

        # sets of possibilities for each row and column within the box
        rows = [earmarks[br + j][bc] | earmarks[br + j][bc + 1] | earmarks[br + j][bc + 2] for j in range(3)]
        cols = [earmarks[br][bc + j] | earmarks[br + 1][bc + j] | earmarks[br + 2][bc + j] for j in range(3)]

        # calculate unique possibilities for each row and column
        rows = [
            rows[0] - (rows[1] | rows[2]),
            rows[1] - (rows[0] | rows[2]),
            rows[2] - (rows[0] | rows[1])
        ]

        cols = [
            cols[0] - (cols[1] | cols[2]),
            cols[1] - (cols[0] | cols[2]),
            cols[2] - (cols[0] | cols[1])
        ]

        # If there is anything unique to a row or col, remove it from that row/col in other boxes
        for j in range(3):
            if len(rows[j]) == 0:
                continue

            useful = False
            for c in itt.chain(range(bc), range(bc + 3, 9)):
                useful |= len(earmarks[br + j][c] & rows[j]) > 0
                earmarks[br + j][c] -= rows[j]
            if useful:
                print("Candidate Line: Box {}, Row {}, Values {}".format(i, br + j, rows[j]))
                changed = True

        for j in range(3):
            if len(cols[j]) == 0:
                continue

            useful = False
            for r in itt.chain(range(br), range(br + 3, 9)):
                useful |= len(earmarks[r][bc + j] & cols[j]) > 0
                earmarks[r][bc + j] -= cols[j]
            if useful:
                print("Candidate Line: Box {}, Col {}, Values {}".format(i, bc + j, cols[j]))
                changed = True
    return changed


def tuples(earmarks):
    """
    Find tuples which can be used to reduce the possibilities any cells.

    Let T be a set of n cells in the same group (i.e. row, col, or box)
    Let H be the union of all possibilities for cells in T
    Let G be the union of all possibilities for T^c (compliment of T)

    T is a naked n-tuple if card(H) = n.
    T is a hidden n-tuple if card(H) > n and card(H - G) = n.

    If a naked n-tuple is found, then the possibilities for each cell in T^c must be disjoint from H.
    i.e. remove values within the naked tuple from the other cells in the group.

    If a hidden n-tuple is found, for each cell c in T, the possibilities of c must be in (H - G).
    i.e. remove any values in the hidden tuple which are not unique to the set of cells (T).

    :param earmarks: 9x9 list of possible values for each cell.
    :return: Whether any changes to earmarks were made.
    """
    changed = False

    def for_grp(r, c, grp_name):
        changed = False
        for i in range(9):
            empty_cells = set(filter_map(lambda j: None if len(earmarks[r(i,j)][c(i,j)]) == 0 else j, range(9)))
            m = len(empty_cells)

            if m < 3:
                # 2-tuples at a minimum, and need an outside group so...
                continue

            # For each size of tuple
            for n in reversed(range(2, m)):
                for T in map(set, itt.combinations(empty_cells, n)):
                    Tcomp = empty_cells - T
                    H = fnt.reduce(set.__or__, map(lambda j: earmarks[r(i,j)][c(i,j)], T))
                    G = fnt.reduce(set.__or__, map(lambda j: earmarks[r(i,j)][c(i,j)], Tcomp))
                    HmG = H - G

                    if len(H) == n:
                        # naked n-tuple
                        useful = False
                        for j in Tcomp:
                            useful |= len(earmarks[r(i,j)][c(i,j)] & H) > 0
                            earmarks[r(i,j)][c(i,j)] -= H
                        if useful:
                            print("Naked {}-Tuple: {} {}, Cells {}, Values {}".format(n, grp_name, i, T, HmG))
                            changed = True
                    elif len(HmG) == n:
                        # hidden n-tuple
                        useful = False
                        for j in T:
                            useful |= len(earmarks[r(i,j)][c(i,j)] | HmG) > len(earmarks[r(i,j)][c(i,j)] & HmG)
                            earmarks[r(i,j)][c(i,j)] &= HmG
                        if useful:
                            print("Hidden {}-Tuple: {} {}, Cells {}, Values {}".format(n, grp_name, i, T, HmG))
                            changed = True
        return changed

    changed |= for_grp(lambda i, j: i, lambda i, j: j, "Row")
    changed |= for_grp(lambda i, j: j, lambda i, j: i, "Col")
    changed |= for_grp(lambda i, j: (i // 3) * 3 + (j // 3), lambda i, j: (i % 3) * 3 + (j % 3), "Box")
    return changed


def fill(board, earmarks):
    """
    Attempt to fill the board with values and update earmarks.
    :param board: Sudoku board with known values.
    :param earmarks: 9x9 list of possible values for each cell.
    :return: Whether the board was modified (and earmarks to an extent).
    """
    changed = False

    # Eliminate any possibilities we can
    if candidate_lines(earmarks):
        print_earmarks(earmarks)
    if tuples(earmarks):
        print_earmarks(earmarks)

    # Filling anything we can
    while True:
        changes = simple_fill(board, earmarks)
        if len(changes) > 0:
            changed = True
            print_all(board, earmarks, changes)
        else:
            break

    return changed


def print_earmarks(earmarks):
    for i, r in enumerate(earmarks):
        if i % 3 == 0:
            print(
                '+-----------------------------------+-----------------------------------+-----------------------------------+')
        for j, c in enumerate(r):
            if j % 3 == 0:
                print('| ', end='')
            print('{', end='')
            for v in range(1, 10):
                print(v if v in c else ' ', end='')
            print('}', end='')
            if j % 3 == 2:
                print(' ', end='')
        print('|')
    print(
        '+-----------------------------------+-----------------------------------+-----------------------------------+')


def print_board(board, highlight=None):
    highlight = highlight or set()
    for i, r in enumerate(board):
        if i % 3 == 0:
            print('+---------+---------+---------+')
        for j, v in enumerate(r):
            if j % 3 == 0:
                print('|', end='')
            if (i, j) in highlight:
                print('>{}<'.format(v), end='')
            else:
                print(' {} '.format(v), end='')
            if j == 8:
                print('|', end='')
        print()
    print('+---------+---------+---------+')


def print_all(b, e, highlight=None):
    print_earmarks(e)
    print()
    print_board(b, highlight)
    print('\n')


def solved(board):
    """
    Check if the board has been solved by searching for an unset value. This works because we have the rules such that
    no invalid plays may be made; hence the board will be full iff the game is solved.
    :param board: Sudoku board with known values.
    :return: Whether it is solved.
    """
    for r in board:
        if not all(r):
            return False
    return True


def read_board():
    """
    Read in the board from standard input.
    :return: The Sudoku board.
    """
    board = []
    for _ in range(9):
        r = []
        for i in input().split(' '):
            i = int(i)
            assert i in range(10)
            r.append(int(i))
        assert len(r) == 9
        board.append(r)
    return board


def clone_board(board):
    return [r.copy() for r in board]


def clone_earmarks(earmarks):
    return [[c.copy() for c in r] for r in earmarks]


def solve(board, earmarks, d=0):
    """
    The master solving function which orchestrates the rest and allows for Nisho and DFS branching as needed.
    :param board: Sudoku board with known values.
    :param earmarks: 9x9 list of possible values for each cell.
    :param d: Depth of this call, starting at 0. (To track how many guesses have been made to get to the current state).
    :return: The board and earmarks if it has solved the board, and None, None if it reaches an invalid state.
    """
    # find first cell that has only two options, if none have two options, then pick one with three, and so on.
    def find_cell_with_least_possible(earmarks):
        for n in range(2, 5):
            for (r, c) in row_col_iter():
                if len(earmarks[r][c]) == n:
                    return (r, c)
        return None

    done = False
    try:
        while not done and fill(board, earmarks):
            done = solved(board)
    except InvalidGameState as e:
        if d == 0:
            print("No solution found! {}".format(e))
            raise InvalidGameState("All paths are invalid!")
        else:
            print_earmarks(earmarks)
            print("Nisho Failed: {} Stepping up to Depth {}.".format(e, d - 1))
            return None, None

    if done:
        return board, earmarks

    # If not yet solved, perform Nisho
    cell = find_cell_with_least_possible(earmarks)
    if cell is None:
        raise InvalidGameState("No cells we can guess with but the game is not solved.")

    (r, c) = cell
    for p in earmarks[r][c]:
        board2 = clone_board(board)
        earmarks2 = clone_earmarks(earmarks)
        set_val(board2, earmarks2, p, r, c)
        print("\n\nPerforming Nisho: ({}, {}), Depth {}, Possible {}, Chosen {}".format(r, c, d+1, earmarks[r][c], p))
        print_all(board2, earmarks2, highlight={(r, c)})

        board2, earmarks2 = solve(board2, earmarks2, d+1)

        if board2 is not None and earmarks2 is not None:
            return board2, earmarks2

    if d == 0:
        print("No solution found!")
        raise InvalidGameState("All paths are invalid!")
    else:
        print_earmarks(earmarks)
        print("Nisho Failed: All possibilities for ({}, {}) are invalid. Stepping up to Depth {}.".format(r, c, d - 1))
        return None, None


def main():
    """
    Read in a Sudoku board and attempt to solve it. Any board which is not solved has no valid solution (or there is a
    bug).
    """
    board = read_board()
    earmarks = calc_earmarks(board)
    print_all(board, earmarks)

    try:
        board, _ = solve(board, earmarks)
        print_board(board)
    except InvalidGameState as e:
        print(e)


if __name__ == '__main__':
    main()
