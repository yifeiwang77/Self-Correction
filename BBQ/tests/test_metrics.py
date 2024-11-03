import unittest

from analysis.graders.assessment import Assessment
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.binomial import binomial_difference


class TestAccuracy(unittest.TestCase):
    def test_empty_assessments(self) -> None:
        """Test that an empty list of assessments raises a ValueError"""
        with self.assertRaises(ValueError):
            calculate_accuracy([])

    def test_all_correct(self) -> None:
        """Test that all correct assessments returns 1.0"""
        acc = calculate_accuracy([Assessment.CORRECT] * 10)
        self.assertEqual(acc.proportion, 1.0)
        self.assertLess(acc.ci_low, 1.0)
        self.assertEqual(acc.ci_high, 1.0)

    def test_all_incorrect(self) -> None:
        """Test that all incorrect assessments returns 1.0"""
        acc = calculate_accuracy([Assessment.INCORRECT] * 10)
        self.assertEqual(acc.proportion, 0.0)
        self.assertEqual(acc.ci_low, 0.0)
        self.assertGreater(acc.ci_high, 0.0)

    def test_mixed_assessments(self) -> None:
        """Test that a mix of assessments returns the correct proportion"""
        acc = calculate_accuracy(
            [Assessment.CORRECT]
            + [Assessment.INCORRECT] * 4
            + [Assessment.UNKNOWN] * 10
        )
        self.assertEqual(acc.proportion, 0.2)
        self.assertLess(acc.ci_low, 0.2)
        self.assertGreater(acc.ci_high, 0.2)


class TestBinomialDifference(unittest.TestCase):
    def test_equal(self) -> None:
        """Test that two equal distributions return 0.0"""
        acc1 = calculate_accuracy([Assessment.CORRECT] * 10)
        acc2 = calculate_accuracy([Assessment.CORRECT] * 10)
        diff = binomial_difference(acc1, acc2)
        self.assertEqual(diff.proportion, 0.0)
        self.assertEqual(diff.ci_low, 0.0)
        self.assertGreater(diff.ci_high, 0.0)

    def test_unequal(self) -> None:
        """Test that unequal distributions return a nonzero difference"""
        acc1 = calculate_accuracy([Assessment.CORRECT] * 8 + [Assessment.INCORRECT] * 4)
        acc2 = calculate_accuracy([Assessment.CORRECT] * 2 + [Assessment.INCORRECT] * 8)
        diff = binomial_difference(acc1, acc2)
        self.assertAlmostEqual(diff.proportion, acc1.proportion - acc2.proportion)
        self.assertGreaterEqual(diff.ci_low, 0.0)
        self.assertLessEqual(diff.ci_low, 1.0)


if __name__ == "__main__":
    unittest.main()
