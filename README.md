# Sudoku Solver
This is a Python3 Sudoku solver which uses set-theory to determine what values must go at each location
in a Sudoku puzzle. Specifically it handles Single Position and Single Candidate (the two easy ones), but
also finds candidate lines, naked n-tuples, and hidden n-tuples. Finally, if all else fails, which only
happens in the hardest of problems, it also supports a DFS approach to Nisho (educated guessing).

By defining each of these checks in terms of set theory, we are able to avoid extra levels of
iteration, and it could be optimized by using bit-wise logic as their are only 9 possible values each
cell can take.

Much of the following logic was determined from examples provided by
[sudokuoftheday](https://www.sudokuoftheday.com/techniques/) which were then converted to be in terms of
set-theory to enable this solver.

### Single Candidate and Single Position
Single Candidate is when there is only one possibility for a cell, so we know that it must be the value,
and Single Position is when there is only one place a value can go within a group.

Let P be the set of possibilities for a cell.
A cell is a single candidate if card(P) = 1.

Let C be a single cell.  
Let P be the possibilities for C.  
Let T be a set of n cells in the same group as C (i.e. row, col, or box and excluding C).  
Let H be the union of all possibilities for cells in T.  
A cell C is a single position if it is a member of a group T such that card(H - P) = 1.

In either the case of Single Position or Single Candidate, we can determine the value of the cell.


### Candidate Lines
If the possibilities for a given number in a group are all in a line on a row or column, then we can
remove those as possibilities from the rest of the row/column.

Let B be a 3x3 box of cells (i.e. one of the sudoku 3x3 boxes).  
Let R be the possibilities for a partial row which is a subset of B (i.e. a 3-long row).  
Let C be the possibilities for a partial column which is a subset of B (i.e. a 3-long column).

Let R1, R2, R3 in B denote all possibilities of partial rows of B.  
Let C1, C2, C3 in B denote all possibilities of partial columns of B.

Then, without loss of generality, the set of candidate lines for a given partial row R1 is
R1 - (R2 | R3), and further, the set of candidate lines for a given partial column C1 is C1 - (C2 | C3).


### N-Tuples
While naked n-tuples define what values outside cells cannot take and hidden n-tuples define what values
inside cells cannot take, they both have a lot of overlap in calculation.

Let T be a set of n cells in the same group (i.e. row, col, or box).  
Let H be the union of all possibilities for cells in T.  
Let G be the union of all possibilities for T^c (compliment of T).

T is a naked n-tuple if card(H) = n.  
T is a hidden n-tuple if card(H) > n and card(H - G) = n.

If a naked n-tuple is found, then the possibilities for each cell in T^c must be disjoint from H.  
i.e. remove values within the naked tuple from the other cells in the group.

If a hidden n-tuple is found, for each cell c in T, the possibilities of c must be in (H - G).  
i.e. remove any values in the hidden tuple which are not unique to the set of cells (T).