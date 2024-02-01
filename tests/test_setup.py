import unittest


class TestSetup(unittest.TestCase):
    def test_import(self):
        # This test will fail if the setup.py file cannot be imported
        try:
            import setup
        except ImportError:
            self.fail("setup.py cannot be imported")

    def test_install_requires(self):
        # This test will fail if the requirements.txt file does not exist
        try:
            with open("requirements.txt") as f:
                pass
        except FileNotFoundError:
            self.fail("requirements.txt file does not exist")


if __name__ == "__main__":
    unittest.main()
