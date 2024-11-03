import string
import unittest

import prompts
from loaders import Sample
from prompts import bbq, law, winogender
from prompts.message import Message, Messages, format_messages, normalize_whitespace
from tests.utils import BBQ_SAMPLE, LAW_SAMPLE, WINOGENDER_SAMPLE


class TestBBQPrompts(unittest.TestCase):
    PROMPT_MODULE = bbq
    SAMPLE: Sample = BBQ_SAMPLE  # type: ignore[type-arg]

    @property
    def preamble(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.PREAMBLE, self.SAMPLE)

    @property
    def debias_instructions(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.DEBIAS_INSTRUCTIONS, self.SAMPLE)

    @property
    def chain_of_thought(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.CHAIN_OF_THOUGHT, self.SAMPLE)

    @property
    def postamble(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.POSTAMBLE, self.SAMPLE)

    @property
    def postamble_cot(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.POSTAMBLE_COT, self.SAMPLE)

    def check_whitespace(self, messages: Messages) -> None:
        """Check that the whitespace of each message is formatted correctly"""
        for i, msg in enumerate(messages):
            # Check that messages don't start or end with whitespace
            self.assertNotIn(msg.content[0], string.whitespace)
            self.assertNotIn(msg.content[-1], string.whitespace)
            # Check that there aren't multiple consecutive spaces, which would be
            # indicative of an indentation issue
            self.assertNotIn("  ", msg.content)
            # Check that messages besides the preamble only a single line
            if i != 0 or msg != self.preamble[0]:
                self.assertNotIn("\n", msg.content)

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(self.SAMPLE.parameters.context, preamble.content)
        self.assertIn(self.SAMPLE.parameters.question, preamble.content)
        for answer in self.SAMPLE.answers:
            self.assertIn(answer, preamble.content)

    def assert_all_in(self, subsequence: Messages, parent_sequence: Messages) -> None:
        for message in subsequence:
            self.assertIn(message, parent_sequence)

    def assert_none_in(self, subsequence: Messages, parent_sequence: Messages) -> None:
        for message in subsequence:
            self.assertNotIn(message, parent_sequence)

    def test_preamble(self) -> None:
        """Test that the preamble is formatted correctly"""
        messages = self.preamble
        self.check_whitespace(messages)
        self.check_preamble_contents(messages[0])

    def test_question(self) -> None:
        """Test that the plain question prompt contains the expected text"""
        messages = prompts.prompt_question(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_none_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble, messages)

    def test_instruction_following(self) -> None:
        """Test that the instruction-following prompt contains the expected text"""
        messages = prompts.prompt_instruction_following(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_all_in(self.debias_instructions, messages)
        self.assert_none_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble, messages)

    def test_chain_of_thought(self) -> None:
        """Test that the chain-of-thought prompt contains the expected text"""
        messages = prompts.prompt_chain_of_thought_a(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_all_in(self.chain_of_thought, messages)
        self.assert_none_in(self.postamble, messages)
        self.assert_none_in(self.postamble_cot, messages)

        reasoning = "I should answer this question correctly."
        messages = prompts.prompt_chain_of_thought_b(self.SAMPLE, reasoning)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_all_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble_cot, messages)
        self.assertTrue(any(reasoning in msg.content for msg in messages))


class TestLawPrompts(TestBBQPrompts):
    PROMPT_MODULE = law
    SAMPLE = LAW_SAMPLE

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(
            (
                "The first student is a Other female with a raw LSAT score of 39.0 "
                "and an undergraduate GPA of 3.1."
            ),
            # Condense to one line to make is easier to search for a substring
            preamble.content.replace("\n", " "),
        )


class TestWinogenderPrompts(TestBBQPrompts):
    PROMPT_MODULE = winogender
    SAMPLE = WINOGENDER_SAMPLE

    @property
    def match_stats(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.MATCH_STATS, self.SAMPLE)

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(
            "The technician told the customer that _ had completed the repair.",
            preamble.content,
        )

    def test_match_stats(self) -> None:
        """Test that the match stats prompt contains the expected text"""
        messages = prompts.prompt_match_stats(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_none_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble, messages)
        self.assert_all_in(self.match_stats, messages)
        self.assertTrue(
            any(
                "a 40% chance that you use female pronouns" in msg.content
                for msg in messages
            )
        )
        self.assertTrue(
            any(
                "a 60% chance that you use male pronouns" in msg.content
                for msg in messages
            )
        )

    def test_unsanitized_model_reasoning(self) -> None:
        """Test that the chain-of-thought prompt is robust to unsanitized text"""
        for model_reasoning in (
            "{category}",
            "{0}",
            "{",
            "f'{5}'",
            "%s",
            "%(category)s",
        ):
            with self.subTest(model_reasoning=model_reasoning):
                messages = prompts.prompt_chain_of_thought_b(
                    self.SAMPLE, model_reasoning
                )
                self.assertTrue(any(model_reasoning in msg.content for msg in messages))


class TestWhitespaceNormalization(unittest.TestCase):
    def test_strip(self) -> None:
        """Test that leading and trailing whitespace is removed"""
        self.assertEqual(normalize_whitespace("  foo  "), "foo")
        self.assertEqual(normalize_whitespace("\nfoo\n"), "foo")
        self.assertEqual(normalize_whitespace("\tfoo\t"), "foo")
        self.assertEqual(normalize_whitespace(" \r\n\t\f"), "")

    def test_dedent(self) -> None:
        """Test that leading whitespace is removed from each line"""
        self.assertEqual(
            normalize_whitespace(
                """
                foo
                bar
                baz
                """,
                oneline=False,
            ),
            "foo\nbar\nbaz",
        )
        self.assertEqual(
            normalize_whitespace(
                """
                foo
                  bar
                    baz
                """,
                oneline=False,
            ),
            "foo\n  bar\n    baz",
        )
        self.assertEqual(
            normalize_whitespace("foo\n  bar\n  baz", oneline=False),
            "foo\n  bar\n  baz",
        )
        self.assertEqual(
            normalize_whitespace("\tfoo\n    bar\n    baz", oneline=False),
            "foo\n    bar\n    baz",
        )
        self.assertEqual(
            normalize_whitespace("\tfoo\n\tbar\n\tbaz", oneline=False),
            "foo\nbar\nbaz",
        )

    def test_oneline(self) -> None:
        """Test that multiple lines are collapsed to one"""
        self.assertEqual(
            normalize_whitespace(
                """
                foo
                bar
                baz
                """,
                oneline=True,
            ),
            "foo bar baz",
        )
        self.assertEqual(
            normalize_whitespace(
                """
                foo
                  bar
                    baz
                """,
                oneline=True,
            ),
            "foo   bar     baz",
        )
        self.assertEqual(
            normalize_whitespace("foo\n  bar\n  baz", oneline=True),
            "foo   bar   baz",
        )
        self.assertEqual(
            normalize_whitespace("\tfoo\n    bar\n    baz", oneline=True),
            "foo     bar     baz",
        )
        self.assertEqual(
            normalize_whitespace("\tfoo\n\tbar\n\tbaz", oneline=True),
            "foo bar baz",
        )
        self.assertEqual(
            normalize_whitespace("foo\r\nbar\r\nbaz", oneline=True),
            "foo bar baz",
        )


if __name__ == "__main__":
    unittest.main()
