#!/usr/bin/env python3

import threading
import time
import unittest

import speech_recognition as sr


class FakeStream:
    """Mimics a PyAudio stream's .read(): returns preset chunks in order,
    then keeps returning a filler chunk (never an empty/EOF read) so the
    listener loop keeps running until the test explicitly stops it.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0
        self._filler_chunk = b"\x00\x00" * 8

    def read(self, _size):
        time.sleep(0.001)  # avoid a tight CPU-spinning loop in tests
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        return self._filler_chunk


class FakeAudioSource(sr.AudioSource):
    def __init__(self, chunks, chunk_size=1024, sample_rate=16000, sample_width=2):
        self.CHUNK = chunk_size
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = sample_width
        self.stream = None
        self._chunks = chunks

    def __enter__(self):
        self.stream = FakeStream(self._chunks)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream = None
        return False


class FakeSegment:
    def __init__(self, audio):
        self.audio = audio


class FakeSegmenter:
    """A scripted stand-in for VadSegmenter: process() returns whatever list
    was queued for that call (default: none), flush() returns a preset
    result (default: None).
    """

    def __init__(self, sample_rate=16000, sample_width=2):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.received_chunks = []
        self.flush_result = None
        self.process_results = []

    def process(self, pcm_bytes):
        self.received_chunks.append(pcm_bytes)
        if self.process_results:
            return self.process_results.pop(0)
        return []

    def flush(self):
        return self.flush_result


class TestListenWithSegmenterInBackground(unittest.TestCase):
    def test_delivers_a_completed_segment_via_callback(self):
        recognizer = sr.Recognizer()
        segmenter = FakeSegmenter()
        segmenter.process_results = [[], [FakeSegment(b"segment-audio")]]
        source = FakeAudioSource([b"chunk1", b"chunk2"])

        received = []
        event = threading.Event()

        def callback(_recognizer, audio):
            received.append(audio)
            event.set()

        stop, _pause, _resume = recognizer.listen_with_segmenter_in_background(source, callback, segmenter)
        try:
            self.assertTrue(event.wait(timeout=2), "callback was not invoked in time")
        finally:
            stop()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].get_raw_data(), b"segment-audio")

    def test_stop_flushes_a_pending_segment_instead_of_dropping_it(self):
        recognizer = sr.Recognizer()
        segmenter = FakeSegmenter()
        segmenter.flush_result = FakeSegment(b"flushed-audio")
        source = FakeAudioSource([b"ab"] * 50)  # never finalizes anything on its own

        received = []

        def callback(_recognizer, audio):
            received.append(audio)

        stop, _pause, _resume = recognizer.listen_with_segmenter_in_background(source, callback, segmenter)
        time.sleep(0.05)  # let the listener thread read a few chunks
        stop()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].get_raw_data(), b"flushed-audio")

    def test_stop_without_a_pending_segment_delivers_nothing_extra(self):
        recognizer = sr.Recognizer()
        segmenter = FakeSegmenter()  # flush_result stays None
        source = FakeAudioSource([b"ab"] * 50)

        received = []
        stop, _pause, _resume = recognizer.listen_with_segmenter_in_background(
            source, lambda _r, audio: received.append(audio), segmenter,
        )
        time.sleep(0.05)
        stop()

        self.assertEqual(received, [])

    def test_callback_energy_fires_per_chunk_read(self):
        recognizer = sr.Recognizer()
        segmenter = FakeSegmenter()
        source = FakeAudioSource([b"ab"] * 50)

        energies = []
        stop, _pause, _resume = recognizer.listen_with_segmenter_in_background(
            source, lambda *_: None, segmenter, callback_energy=energies.append,
        )
        time.sleep(0.05)
        stop()

        self.assertGreater(len(energies), 0)

    def test_stop_terminates_the_background_thread_promptly(self):
        recognizer = sr.Recognizer()
        segmenter = FakeSegmenter()
        source = FakeAudioSource([b"ab"] * 5000)

        stop, _pause, _resume = recognizer.listen_with_segmenter_in_background(source, lambda *_: None, segmenter)
        time.sleep(0.02)

        start = time.time()
        stop(wait_for_stop=True)
        elapsed = time.time() - start

        self.assertLess(elapsed, 2.0, "stop() should not hang waiting for the listener thread")


if __name__ == "__main__":
    unittest.main()
