# -*- coding: utf-8 -*-
import pytest
from puzzlelib import Puzzle

class TestPuzzlelib:

    def test_new_puzzle(self):
        pz = Puzzle(3, 3)
        assert(pz.size_h == 3 and pz.size_v == 3)
        assert(len(pz.start) == 9)
        assert(len(pz.goal) == 9)
        # default for wrong sizes
        pz = Puzzle(0, 3)
        assert (pz.size_h == 4 and pz.size_v == 4)
        pz = Puzzle(4, 11)
        assert(pz.size_h == 4 and pz.size_v == 4)
        assert(len(pz.start) == 16)
        assert(len(pz.goal) == 16)

    def test_generate_puzzle(self):
        pz = Puzzle(4,4)
        pz.generate()
        assert(len(pz.puzzle) == 4*4)
        assert(len(pz.start) == 4*4)

    def test_set_puzzle(self):
        pz = Puzzle(2, 2)
        assert(pz.set_puzzle([3, 1, 2, 0]))
        assert(pz.puzzle == [3, 1, 2, 0])
        assert(pz.start == [3, 1, 2, 0])
        assert(pz.goal == [1, 2, 3, 0])
        assert(not pz.set_puzzle([3, 2, 1, 0]))
        pz = Puzzle(3, 3)
        assert(not pz.set_puzzle([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))

    def test_search_all_sets(self):
        pz = Puzzle(2, 2)
        assert(pz.set_puzzle([3, 1, 2, 0]))
        sets = pz.search_all_sets([3, 1, 2, 0])
        assert(len(sets) == 4)
        assert(sets[0] == [3, 0, 2, 1])
        assert(sets[1] == [3, 1, 2, -1])
        assert(sets[2] == [3, 1, 0, 2])
        assert(sets[3] == [3, 1, 2, -1])
        pz = Puzzle(3, 4)
        sets = pz.search_all_sets([8, 10, 6, 2, 9, 4, 11, 0, 1, 3, 5, 7])
        assert(len(sets) == 4)
        assert(sets[0] == [8, 10, 6, 2, 0, 4, 11, 9, 1, 3, 5, 7])
        assert(sets[1] == [8, 10, 6, 2, 9, 4, 11, 5, 1, 3, 0, 7])
        assert(sets[2] == [8, 10, 6, 2, 9, 4, 0, 11, 1, 3, 5, 7])
        assert(sets[3] == [8, 10, 6, 2, 9, 4, 11, 1, 0, 3, 5, 7])

    def test_search_solution(self):
        pz = Puzzle(3, 3)
        pz.generate()
        assert(len(pz.puzzle) == 3 * 3)
        assert(len(pz.start) == 3*3)
        assert(pz.goal == [1, 2, 3, 4, 5, 6, 7, 8, 0])
        maps = pz.search_solution()
        assert(len(maps) > 2)
        assert(maps[-1].set.position == pz.start)
        assert(maps[0].set.position == pz.goal)

