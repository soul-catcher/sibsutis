import unittest

import utils


class SplitByCommasTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by('A, B, C', ','))

    def test_without_space(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by('A,B,C', ','))

    def test_with_chaotic_space(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by(' A   , B ,   C  ', ','))

    def test_without_commas(self):
        self.assertEqual(['A  B C'], utils.split_by('A  B C', ','))

    def test_zero_length(self):
        self.assertEqual([], utils.split_by('', ','))

    def test_only_spaces(self):
        self.assertEqual([], utils.split_by('  ', ','))

    def test_word(self):
        self.assertEqual(['test'], utils.split_by('test', ','))

    def test_several_words_without_commas(self):
        self.assertEqual(['one two three'], utils.split_by('  one two three ', ','))

    def test_several_words_with_commas(self):
        self.assertEqual(['one', 'two', 'three'], utils.split_by('one, two,three', ','))

    def test_chaotic_commas(self):
        self.assertEqual(['A', 'B', 'C'], utils.split_by(',,A,,,B,C,', ','))

    def test_only_comma(self):
        self.assertEqual([], utils.split_by(',', ','))


class ParseRulesTest(unittest.TestCase):
    def test_simple(self):
        rules = 'A -> aAa'
        expected = {'A': ['aAa']}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_lambda(self):
        rules = 'A -> aAa | @'
        expected = {'A': ['aAa', '']}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_lambda_left(self):
        rules = 'A -> @ | aAa'
        expected = {'A': ['', 'aAa']}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_without_spaces(self):
        rules = 'A->aAa|@'
        expected = {'A': ['aAa', '']}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_chaotic_spaces(self):
        rules = '   AB  C ->    a A    a  |    @  '
        expected = {'AB  C': ['a A    a', '']}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_empty_rules(self):
        rules = ''
        expected = {}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_multiline_rules(self):
        rules = '''
        A -> aAa
        
        B -> bBb
        '''
        expected = {'A': ['aAa'], 'B': ['bBb']}
        self.assertEqual(expected, utils.parse_rules(rules, '@'))

    def test_without_arrow(self):
        with self.assertRaises(utils.WrongRulesException):
            utils.parse_rules('A aAa', '@')

    def test_wrong_place_arrow1(self):
        with self.assertRaises(utils.WrongRulesException):
            utils.parse_rules('-> A aAa', '@')

    def test_wrong_place_arrow2(self):
        with self.assertRaises(utils.WrongRulesException):
            utils.parse_rules('A aAa -> ', '@')

if __name__ == '__main__':
    unittest.main()
