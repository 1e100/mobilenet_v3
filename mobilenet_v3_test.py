import doctest
import unittest


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite("mobilenet_v3"))
    return tests


if __name__ == "__main__":
    unittest.main()
