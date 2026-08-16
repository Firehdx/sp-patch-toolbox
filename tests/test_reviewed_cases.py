import unittest

from sp_patch_toolbox.profiles.reviewed_cases import REVIEWED_CASE_GROUPS, reviewed_case_values


class ReviewedCaseTests(unittest.TestCase):
    def test_reviewed_case_catalogue_matches_compatibility_values(self):
        values = reviewed_case_values()
        self.assertEqual(set(values), set(REVIEWED_CASE_GROUPS))
        self.assertTrue(all(values[group] for group in values))
