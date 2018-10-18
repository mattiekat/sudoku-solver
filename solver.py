import itertools as itt
import functools as fnt

# Largely inspired by https://www.sudokuoftheday.com/techniques/
VALUES = set([x + 1 for x in range(9)])


def filter_map(fn, it):
    """
    Map and filter values. If a value is mapped to None, then it will be filtered.
    :param fn: Function used to map values.
    :param it: Values over which the function should run.
    :return: A filter_map iterator which yeilds non-None values mapped by the function.
    """
    return filter(lambda x: x is not None, map(fn, it))


def calc_earmarks(board):
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

    # TODO: Consider using u16 and bit-logic instead of sets to make operations more efficient
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
    board[r][c] = newval
    update_earmarks(earmarks, newval, r, c)


def simple_fill(board, earmarks):
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
    for (r, c) in map(lambda i: (i // 9, i % 9), range(9**2)):
        if board[r][c] != 0:
            continue

        possible = earmarks[r][c]
        assert len(possible) > 0

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
            assert len(p) <= 1
            if len(p) == 1:
                val = p.pop()
                print("Single Position ({}): ({}, {}), Val {}".format(['Col', 'Row', 'Box'][i], r, c, val))
                set_val(board, earmarks, val, r, c)
                changed.add((r, c))
                break
    return changed


def candidate_lines(earmarks):
    # If the possibilities for a given number in a group are all in a line on a row or column, then we can remove
    # those as possibilities from the rest of the row/column
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
            print("Candidate Line: Box {}, Row {}, Values {}".format(i, br + j, rows[j]))
            for c in itt.chain(range(bc), range(bc + 3, 9)):
                earmarks[br + j][c] -= rows[j]

        for j in range(3):
            if len(cols[j]) == 0:
                continue
            print("Candidate Line: Box {}, Col {}, Values {}".format(i, bc + j, cols[j]))
            for r in itt.chain(range(br), range(br + 3, 9)):
                earmarks[r][bc + j] -= cols[j]


def tuples(earmarks):
    # Let T be a set of n cells in the same group (i.e. row, col, or box)
    # Let H be the union of all possibilities for cells in T
    # Let G be the union of all possibilities for T^c (compliment of T)
    #
    # T is a naked n-tuple if card(H) = n.
    # T is a hidden n-tuple if card(H) > n and card(H - G) = n.
    #
    # If a naked n-tuple is found, then the possibilities for each cell in T^c must be disjoint from H.
    # i.e. remove values within the naked tuple from the other cells in the group.
    #
    # If a hidden n-tuple is found, for each cell c in T, the possibilities of c must be in (H - G).
    # i.e. remove any values in the hidden tuple which are not unique to the set of cells (T).

    # For each row
    for r in range(9):
        empty_cells = set(filter_map(lambda c: c if len(earmarks[r][c]) > 0 else None, range(9)))
        m = len(empty_cells)

        if m < 3:
            # 2-tuples at a minimum, and need an outside group so...
            continue

        # For each size of tuple
        for n in reversed(range(2, m)):
            for T in map(lambda t: set(t), itt.combinations(empty_cells, n)):
                Tcomp = empty_cells - T
                H = fnt.reduce(lambda A, B: A | B, map(lambda c: earmarks[r][c], T))
                G = fnt.reduce(lambda A, B: A | B, map(lambda c: earmarks[r][c], Tcomp))
                HmG = H - G

                if len(H) == n:
                    # naked n-tuple
                    print("Naked {}-Tuple: Row {}, Cells {}, Values {}".format(n, r, T, H))
                    for c in Tcomp:
                        earmarks[r][c] -= H
                elif len(HmG) == n:
                    # hidden n-tuple
                    print("Hidden {}-Tuple: Row {}, Cells {}, Values {}".format(n, r, T, HmG))
                    for c in T:
                        earmarks[r][c] &= HmG

    # For each column
    for c in range(9):
        empty_cells = set(filter_map(lambda r: None if len(earmarks[r][c]) == 0 else r, range(9)))
        m = len(empty_cells)

        if m < 3:
            # 2-tuples at a minimum, and need an outside group so...
            continue

        # For each size of tuple
        for n in reversed(range(2, m)):
            for T in map(lambda t: set(t), itt.combinations(empty_cells, n)):
                Tcomp = empty_cells - T
                H = fnt.reduce(lambda A, B: A | B, map(lambda r: earmarks[r][c], T))
                G = fnt.reduce(lambda A, B: A | B, map(lambda r: earmarks[r][c], Tcomp))
                HmG = H - G

                if len(H) == n:
                    # naked n-tuple
                    print("Naked {}-Tuple: Col {}, Cells {}, Values {}".format(n, c, T, H))
                    for r in Tcomp:
                        earmarks[r][c] -= H
                elif len(HmG) == n:
                    # hidden n-tuple
                    print("Hidden {}-Tuple: Col {}, Cells {}, Values {}".format(n, c, T, HmG))
                    for r in T:
                        earmarks[r][c] &= HmG

    # For each box
    for i in range(9):
        br = (i // 3) * 3
        bc = (i % 3) * 3

        def r(j):
            return br + (j // 3)

        def c(j):
            return bc + (j % 3)

        empty_cells = set(filter_map(lambda j: None if len(earmarks[r(j)][c(j)]) == 0 else j, range(9)))
        m = len(empty_cells)

        if m < 3:
            # 2-tuples at a minimum, and need an outside group so...
            continue

        # For each size of tuple
        for n in reversed(range(2, m)):
            for T in map(set, itt.combinations(empty_cells, n)):
                Tcomp = empty_cells - T
                H = fnt.reduce(set.__or__, map(lambda j: earmarks[r(j)][c(j)], T))
                G = fnt.reduce(set.__or__, map(lambda j: earmarks[r(j)][c(j)], Tcomp))
                HmG = H - G

                if len(H) == n:
                    # naked n-tuple
                    print("Naked {}-Tuple: Box {}, Cells {}, Values {}".format(n, i, T, HmG))
                    for j in Tcomp:
                        earmarks[r(j)][c(j)] -= H
                elif len(HmG) == n:
                    # hidden n-tuple
                    print("Hidden {}-Tuple: Box {}, Cells {}, Values {}".format(n, i, T, HmG))
                    for j in T:
                        earmarks[r(j)][c(j)] &= HmG


def fill(board, earmarks):
    changed = False

    # Eliminate any possibilities we can
    candidate_lines(earmarks)
    print_earmarks(earmarks)
    tuples(earmarks)
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
    for r in board:
        for c in r:
            if c == 0:
                return False
    return True


def main():
    board = []
    for _ in range(9):
        r = []
        for i in input().split(' '):
            i = int(i)
            assert i in range(10)
            r.append(int(i))
        assert len(r) == 9
        board.append(r)

    earmarks = calc_earmarks(board)
    print_all(board, earmarks)

    while not solved(board) and fill(board, earmarks):
        pass

    if solved(board):
        print_board(board)
    else:
        print_all(board, earmarks)


if __name__ == '__main__':
    main()
