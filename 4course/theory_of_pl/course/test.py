import unittest
import utils
import grammar


class SplitByCommasTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by_commas('A, B, C'))

    def test_without_space(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by_commas('A,B,C'))

    def test_with_chaotic_space(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by_commas(' A   , B ,   C  '))

    def test_without_commas(self):
        self.assertEqual(['A  B C'], utils.split_by_commas('A  B C'))

    def test_zero_length(self):
        self.assertEqual([], utils.split_by_commas(''))

    def test_only_spaces(self):
        self.assertEqual([], utils.split_by_commas('  '))

    def test_word(self):
        self.assertEqual(['test'], utils.split_by_commas('test'))

    def test_several_words_without_commas(self):
        self.assertEqual(['one two three'], utils.split_by_commas('  one two three '))

    def test_several_words_with_commas(self):
        self.assertEqual(['one', 'two', 'three'], utils.split_by_commas('one, two,three'))

    def test_chaotic_commas(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by_commas(',,A,,,B,C,'))

    def test_only_comma(self):
        self.assertEqual([], utils.split_by_commas(','))


if __name__ == '__main__':
    unittest.main()
