#!/usr/bin/env python3
"""Tests for ``Recognizer.recognize_google``'s ``join_all_results`` option.

The ``www.google.com/speech-api/v2/recognize`` endpoint's response is one or
more newline-separated JSON blocks. When the uploaded clip spans more than
one perceived utterance (e.g. it contains an internal pause, as happens when
a caller merges several VAD segments into one clip before sending), the
endpoint can return a separate non-empty ``result`` block per utterance.
The historical implementation used only the first non-empty block and
discarded the rest, silently dropping later utterances' text. These tests
mock the network call so they run offline and cover the new
``join_all_results`` opt-in without depending on the real endpoint.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import speech_recognition as sr


def _response_with_lines(*result_lines):
    """Builds a fake urlopen response whose .read() yields the given
    newline-joined JSON blocks (each ``result_lines[i]`` is the raw ``result``
    list for that line, e.g. [] for an empty/interim line).
    """
    body = "\n".join(json.dumps({"result": line}) for line in result_lines)
    response = MagicMock()
    response.read.return_value = body.encode("utf-8")
    return response


class TestRecognizeGoogleJoinAllResults(unittest.TestCase):
    def setUp(self):
        self.recognizer = sr.Recognizer()
        # get_flac_data does real audio conversion work we don't care about
        # here; stub it so tests don't need a real AudioData payload.
        self.audio_data = MagicMock(spec=sr.AudioData)
        self.audio_data.sample_rate = 16000
        self.audio_data.get_flac_data.return_value = b"fake-flac-bytes"

    def _alternative(self, transcript, confidence=None):
        alt = {"transcript": transcript}
        if confidence is not None:
            alt["confidence"] = confidence
        return alt

    def test_default_behavior_uses_only_the_first_non_empty_block(self):
        """Historical behavior must be unchanged when join_all_results is not
        passed: later non-empty blocks are ignored."""
        response = _response_with_lines(
            [],
            [{"alternative": [self._alternative("first utterance")]}],
            [{"alternative": [self._alternative("second utterance")]}],
        )
        with patch("speech_recognition.urlopen", return_value=response):
            text = self.recognizer.recognize_google(self.audio_data)

        self.assertEqual(text, "first utterance")

    def test_join_all_results_concatenates_every_non_empty_block(self):
        """The actual fix: with join_all_results=True, later utterances
        within the same uploaded clip are no longer silently dropped."""
        response = _response_with_lines(
            [],
            [{"alternative": [self._alternative("first utterance")]}],
            [],
            [{"alternative": [self._alternative("second utterance")]}],
        )
        with patch("speech_recognition.urlopen", return_value=response):
            text = self.recognizer.recognize_google(self.audio_data, join_all_results=True)

        self.assertEqual(text, "first utterance second utterance")

    def test_join_all_results_averages_confidence_across_blocks(self):
        response = _response_with_lines(
            [{"alternative": [self._alternative("hello", confidence=0.9)]}],
            [{"alternative": [self._alternative("world", confidence=0.5)]}],
        )
        with patch("speech_recognition.urlopen", return_value=response):
            text, confidence = self.recognizer.recognize_google(
                self.audio_data, join_all_results=True, with_confidence=True
            )

        self.assertEqual(text, "hello world")
        self.assertAlmostEqual(confidence, 0.7)

    def test_join_all_results_single_block_matches_default_behavior(self):
        """When there is genuinely only one utterance (the common case),
        join_all_results=True must not change the result at all."""
        response = _response_with_lines([{"alternative": [self._alternative("only one")]}])
        with patch("speech_recognition.urlopen", return_value=response):
            default_text = self.recognizer.recognize_google(self.audio_data)
        with patch("speech_recognition.urlopen", return_value=response):
            joined_text = self.recognizer.recognize_google(self.audio_data, join_all_results=True)

        self.assertEqual(default_text, joined_text)

    def test_join_all_results_skips_blocks_without_a_usable_transcript(self):
        """A non-empty result block that has no alternatives (or no
        transcript in its best alternative) is skipped rather than raising,
        as long as at least one other block is usable."""
        response = _response_with_lines(
            [{"alternative": [self._alternative("usable")]}],
            [{"alternative": []}],
        )
        with patch("speech_recognition.urlopen", return_value=response):
            text = self.recognizer.recognize_google(self.audio_data, join_all_results=True)

        self.assertEqual(text, "usable")

    def test_no_usable_result_at_all_raises_unknown_value_error(self):
        response = _response_with_lines([], [])
        with patch("speech_recognition.urlopen", return_value=response):
            with self.assertRaises(sr.UnknownValueError):
                self.recognizer.recognize_google(self.audio_data, join_all_results=True)

    def test_show_all_with_join_all_results_returns_every_raw_block(self):
        response = _response_with_lines(
            [{"alternative": [self._alternative("first")]}],
            [{"alternative": [self._alternative("second")]}],
        )
        with patch("speech_recognition.urlopen", return_value=response):
            raw = self.recognizer.recognize_google(
                self.audio_data, join_all_results=True, show_all=True
            )

        self.assertEqual(len(raw), 2)

    def test_show_all_without_join_all_results_returns_only_the_first_block(self):
        response = _response_with_lines(
            [{"alternative": [self._alternative("first")]}],
            [{"alternative": [self._alternative("second")]}],
        )
        with patch("speech_recognition.urlopen", return_value=response):
            raw = self.recognizer.recognize_google(self.audio_data, show_all=True)

        self.assertEqual(raw, {"alternative": [self._alternative("first")]})


if __name__ == "__main__":
    unittest.main()
