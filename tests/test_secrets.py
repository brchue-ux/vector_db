"""Secret-shaped-string filter (report §13.1, failure mode F9).

Every credential below is invented for this test. Nothing here came from the
corpus.
"""

import unittest

from vdb.secrets import redact


class Redaction(unittest.TestCase):
    def assert_gone(self, text, needle, kind=None):
        out, counts = redact(text)
        self.assertNotIn(needle, out)
        self.assertIn("[REDACTED-", out)
        if kind:
            self.assertIn(kind, counts, f"expected {kind}, got {counts}")
        return out, counts

    def test_oauth_callback_code(self):
        # F9's own example shape: a pasted callback URL with a live auth code.
        url = "http://localhost:8080/callback?state=abc&code=4/0AeanS0bQq7xVv9Zk2Lm"
        out, _ = self.assert_gone(url, "4/0AeanS0bQq7xVv9Zk2Lm", "URL-SECRET")
        self.assertIn("code=", out)  # the shape stays readable
        self.assertIn("localhost:8080/callback", out)

    def test_bearer_header(self):
        self.assert_gone(
            "Authorization: Bearer abcDEF123456ghiJKL789mnoPQR", "abcDEF123456ghi", "BEARER"
        )

    def test_aws_access_key(self):
        self.assert_gone("export AWS=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE", "AWS-KEY")

    def test_github_token(self):
        self.assert_gone(
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz", "ghp_1234567890", "GITHUB-TOKEN"
        )

    def test_api_key_prefix(self):
        self.assert_gone("sk-ant-api03-Aa1Bb2Cc3Dd4Ee5Ff6Gg7", "sk-ant-api03-Aa1", "API-KEY")

    def test_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
        self.assert_gone(f"token: {jwt}", jwt, "JWT")

    def test_private_key_block(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEAxyz\nabc\n"
            "-----END RSA PRIVATE KEY-----"
        )
        self.assert_gone(f"here it is:\n{pem}\nthanks", "MIIEowIBAAKCAQEAxyz", "PRIVATE-KEY")

    def test_assigned_password(self):
        out, counts = self.assert_gone(
            'DB_PASSWORD="hunter2hunter2"', "hunter2hunter2", "ASSIGNED-SECRET"
        )
        self.assertIn("DB_PASSWORD", out)

    def test_long_high_entropy_token(self):
        # F9 measured unbroken tokens of >=45 characters as the tell.
        tok = "aB3xQ7zP1mK9wR4tY6uE2iO8sD5fG0hJ7kL3nM1pV9cX4bN6zQ"
        self.assert_gone(f"paste: {tok}", tok, "LONG-TOKEN")

    def test_long_ordinary_words_are_kept(self):
        text = "antidisestablishmentarianism " * 4
        out, counts = redact(text)
        self.assertEqual(out, text)
        self.assertEqual(counts, {})

    def test_long_file_path_is_kept(self):
        path = "/home/user/projects/some-fairly-long-directory-name/and/another/one/file.py"
        out, counts = redact(f"see {path}")
        self.assertIn(path, out)
        self.assertEqual(counts, {})

    def test_ordinary_prose_untouched(self):
        text = "The build failed because the linker could not find libssl."
        self.assertEqual(redact(text), (text, {}))

    def test_counts_are_reported(self):
        _, counts = redact("AKIAIOSFODNN7EXAMPLE and AKIAIOSFODNN7EXAMPLB")
        self.assertEqual(counts["AWS-KEY"], 2)

    def test_empty(self):
        self.assertEqual(redact(""), ("", {}))


if __name__ == "__main__":
    unittest.main()
