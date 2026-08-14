"""Cleaning rules (report §1.1, §7). All fixtures are constructed here; no real
transcript text is ever committed to this repo."""

import unittest

from vdb.clean import CAVEAT, candidate_lines, strip_injected, trim_boilerplate


class StripInjected(unittest.TestCase):
    def test_system_reminder_removed(self):
        text = "Please fix the parser.\n<system-reminder>\nbe helpful\n</system-reminder>\nThanks."
        self.assertEqual(strip_injected(text), "Please fix the parser.\n\nThanks.")

    def test_multiline_and_repeated_blocks(self):
        text = (
            "<system-reminder>a</system-reminder>one"
            "<system-reminder>\nb\nc\n</system-reminder>two"
        )
        self.assertEqual(strip_injected(text), "onetwo")

    def test_command_echo_blocks_removed(self):
        text = (
            "<command-name>/loop</command-name>"
            "<command-args>5m</command-args>"
            "<local-command-stdout>lots of output</local-command-stdout>"
            "what did that do?"
        )
        self.assertEqual(strip_injected(text), "what did that do?")

    def test_bash_blocks_removed(self):
        text = "<bash-input>ls -la</bash-input><bash-stdout>a b c</bash-stdout>ok"
        self.assertEqual(strip_injected(text), "ok")

    def test_task_notification_and_markers(self):
        text = (
            "<task-notification>agent foo finished</task-notification>\n"
            "[Request interrupted by user for tool use]\n"
            "[SYSTEM NOTIFICATION - NOT USER INPUT]\n"
            "real words here"
        )
        self.assertEqual(strip_injected(text), "real words here")

    def test_message_that_is_only_machinery_becomes_empty(self):
        text = "<system-reminder>x</system-reminder>\n<task-notification>y</task-notification>\n"
        self.assertEqual(strip_injected(text), "")

    def test_nested_blocks_removed_to_fixed_point(self):
        text = "<command-message><system-reminder>x</system-reminder></command-message>kept"
        self.assertEqual(strip_injected(text), "kept")

    def test_unterminated_block_does_not_leak(self):
        text = "kept\n<system-reminder>\nthis block never closes and runs to the end"
        self.assertEqual(strip_injected(text), "kept")

    def test_caveat_paragraph_dropped(self):
        text = f"{CAVEAT} and so on.\n\nthe actual question"
        self.assertEqual(strip_injected(text), "the actual question")

    def test_blank_line_runs_collapsed(self):
        self.assertEqual(strip_injected("a\n\n\n\n\nb"), "a\n\nb")

    def test_plain_prose_is_untouched(self):
        text = "Why did the build fail on main?\n\nIt printed a linker error."
        self.assertEqual(strip_injected(text), text)

    def test_empty_input(self):
        self.assertEqual(strip_injected(""), "")


class BoilerplateTrim(unittest.TestCase):
    LINE = "You are a crewmate: an autonomous worker agent managed by firstmate."

    def test_candidate_lines_ignores_short_lines(self):
        text = f"---\nshort\n{self.LINE}"
        self.assertEqual(candidate_lines(text), {self.LINE})

    def test_invariant_line_trimmed_message_survives(self):
        text = f"{self.LINE}\n\n# Task\nBuild the retrieval system."
        out = trim_boilerplate(text, {self.LINE})
        self.assertNotIn(self.LINE, out)
        # §7/§11.5: the templated message itself must NOT be deleted - the task
        # body inside it is the best summary of the session that exists.
        self.assertIn("Build the retrieval system.", out)
        self.assertIn("# Task", out)

    def test_trim_is_a_no_op_without_a_boilerplate_set(self):
        text = f"{self.LINE}\n\nbody"
        self.assertEqual(trim_boilerplate(text, set()), text)

    def test_only_exact_lines_are_trimmed(self):
        text = f"{self.LINE} plus an extra clause\n\nbody"
        self.assertIn("plus an extra clause", trim_boilerplate(text, {self.LINE}))


if __name__ == "__main__":
    unittest.main()
