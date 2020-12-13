import unittest

import utils
from grammar import Grammar


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


class CanonGrammarTest(unittest.TestCase):
    def test_find_non_child_free(self):
        grammar = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S'],
            {
                'S': ['aAB', 'E'],
                'A': ['aA', 'bB'],
                'B': ['ACb', 'b'],
                'C': ['A', 'bA', 'cC', 'aE'],
                'D': ['a', 'c', 'Fb'],
                'E': ['cE', 'aE', 'Eb', 'ED', 'FG'],
                'F': ['BC', 'EC', 'AC'],
                'G': ['Ga', 'Gb']
            },
            'S'
        )
        self.assertSetEqual({'E', 'G'}, grammar.find_child_free_non_terms())

    def test_remove_rules1(self):
        grammar = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S'],
            {
                'S': ['aAB', 'E'],
                'A': ['aA', 'bB'],
                'B': ['ACb', 'b'],
                'C': ['A', 'bA', 'cC', 'aE'],
                'D': ['a', 'c', 'Fb'],
                'E': ['cE', 'aE', 'Eb', 'ED', 'FG'],
                'F': ['BC', 'EC', 'AC'],
                'G': ['Ga', 'Gb']
            },
            'S'
        )
        grammar_expected = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'D', 'F', 'S'],
            {
                'S': ['aAB'],
                'A': ['aA', 'bB'],
                'B': ['ACb', 'b'],
                'C': ['A', 'bA', 'cC'],
                'D': ['a', 'c', 'Fb'],
                'F': ['BC', 'AC'],
            },
            'S'
        )
        self.assertEqual(grammar_expected, grammar.remove_rules({'E', 'G'}))

    def test_remove_rules2(self):
        grammar = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'D', 'F', 'S'],
            {
                'S': ['aAB'],
                'A': ['aA', 'bB'],
                'B': ['ACb', 'b'],
                'C': ['A', 'bA', 'cC'],
                'D': ['a', 'c', 'Fb'],
                'F': ['BC', 'AC'],
            },
            'S'
        )
        grammar_expected = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'S'],
            {
                'S': ['aAB'],
                'A': ['aA', 'bB'],
                'B': ['ACb', 'b'],
                'C': ['A', 'bA', 'cC'],
            },
            'S'
        )
        self.assertEqual(grammar_expected, grammar.remove_rules({'D', 'F'}))

    def test_find_unreachable_rules(self):
        grammar = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'D', 'F', 'S'],
            {
                'S': ['aAB'],
                'A': ['aA', 'bB'],
                'B': ['ACb', 'b'],
                'C': ['A', 'bA', 'cC'],
                'D': ['a', 'c', 'Fb'],
                'F': ['BC', 'AC'],
            },
            'S'
        )
        self.assertSetEqual({'D', 'F'}, grammar.find_unreachable_rules())

    def test_remove_empty_rules(self):
        grammar = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'S'],
            {
                'S': ['AaB', 'aB', 'cC'],
                'A': ['AB', 'a', 'b', 'B'],
                'B': ['Ba', ''],
                'C': ['AB', 'c']
            },
            'S'
        )
        grammar_expected = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'S'],
            {
                'S': ['AaB', 'cC', 'aB', 'Aa', 'a', 'c'],
                'A': ['AB', 'b', 'a', 'B'],
                'B': ['a', 'Ba'],
                'C': ['AB', 'c', 'A', 'B']
            },
            'S'
        )
        self.assertEqual(grammar_expected, grammar.remove_empty_rules())

    def test_remove_chain_rules(self):
        grammar = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'S'],
            {
                'S': ['AaB', 'cC', 'aB', 'Aa', 'a', 'c'],
                'A': ['AB', 'b', 'a', 'B'],
                'B': ['a', 'Ba'],
                'C': ['AB', 'c', 'A', 'B']
            },
            'S'
        )
        grammar_expected = Grammar(
            ['a', 'b', 'c'],
            ['A', 'B', 'C', 'S'],
            {
                'S': ['AaB', 'cC', 'aB', 'Aa', 'a', 'c'],
                'A': ['AB', 'b', 'a', 'Ba'],
                'B': ['a', 'Ba'],
                'C': ['AB', 'c', 'a', 'Ba', 'b']
            },
            'S'
        )
        self.assertEqual(grammar_expected, grammar.remove_chain_rules())


if __name__ == '__main__':
    unittest.main()
