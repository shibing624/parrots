# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Streaming TTS Demo with Real-time Audio Playback
"""
import argparse
import queue
import sys
import time
import numpy as np
from loguru import logger

sys.path.append('..')
import parrots
from parrots.tts import TextToSpeech

parrots_path = parrots.__path__[0]
sys.path.append(parrots_path)


class AudioPlayer:
    """Real-time audio player using sounddevice OutputStream for continuous playback"""
    
    def __init__(self, sample_rate=32000, buffer_size=20):
        self.sample_rate = sample_rate
        self.backend = None
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.is_playing = False
        # Keep an internal pending buffer to avoid re-enqueuing leftovers (prevents overlap)
        self._pending = np.zeros(0, dtype=np.float32)
        
        # Try to initialize sounddevice backend
        try:
            import sounddevice as sd
            self.backend = 'sounddevice'
            self.sd = sd
            logger.info("Using sounddevice OutputStream for continuous audio playback")
        except ImportError:
            logger.warning("sounddevice not available. Audio will be saved but not played.")
            self.backend = None
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback function for OutputStream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Always initialize output with silence
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)
        filled = 0
        
        try:
            while filled < frames:
                # If no pending samples, fetch next buffer from queue
                if self._pending.size == 0:
                    try:
                        data = self.audio_queue.get_nowait()
                    except queue.Empty:
                        # No data available now: leave remaining as silence
                        break
                    
                    if data is None:  # Stop signal
                        self.is_playing = False
                        # Leave remaining as silence and stop stream
                        raise self.sd.CallbackStop
                    
                    # Ensure dtype/shape
                    if not isinstance(data, np.ndarray):
                        data = np.asarray(data, dtype=np.float32)
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    data = data.reshape(-1)  # mono
                    self._pending = data
                
                # Consume from pending
                need = frames - filled
                take = min(need, self._pending.size)
                if take > 0:
                    outdata[filled:filled + take, 0] = self._pending[:take]
                    self._pending = self._pending[take:]
                    filled += take
                else:
                    # Nothing pending (shouldn't happen due to checks), break to avoid tight loop
                    break
        except self.sd.CallbackStop:
            # Propagate stop to sounddevice
            raise
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            # On error, output silence for this block
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
    
    def start(self):
        """Start the audio stream"""
        if self.backend == 'sounddevice' and self.stream is None:
            self.is_playing = True
            self.stream = self.sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=1024,  # Small block size for low latency
            )
            self.stream.start()
            logger.debug("Audio stream started")
    
    def write(self, audio_chunk):
        """Write audio chunk to playback queue"""
        if self.backend == 'sounddevice' and self.is_playing:
            if audio_chunk is None:
                return
            # Ensure audio is float32 mono 1-D
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            audio_chunk = audio_chunk.reshape(-1)
            
            # Put audio data into queue (blocking if queue is full)
            self.audio_queue.put(audio_chunk)
    
    def close(self):
        """Close audio stream"""
        if self.backend == 'sounddevice' and self.stream:
            # Send stop signal
            try:
                self.audio_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            
            # Wait a moment for callback to drain
            time.sleep(0.1)
            
            # Stop and close stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            # Reset state
            self._pending = np.zeros(0, dtype=np.float32)
            self.is_playing = False
            logger.debug("Audio stream closed")


def stream_tts_with_playback(
    tts_model,
    text,
    text_language="auto",
    stream_chunk_size=20,
    save_path=None,
    play_audio=True
):
    """
    Stream TTS generation with real-time playback using incremental decoding
    
    Args:
        tts_model: TextToSpeech model instance
        text: str, text to synthesize
        text_language: str, language of text
        stream_chunk_size: int, number of semantic tokens per chunk
        save_path: str, path to save complete audio (optional)
        play_audio: bool, whether to play audio in real-time
    """
    logger.info(f"Starting streaming TTS for text: {text}")
    logger.info(f"Text language: {text_language}, Chunk size: {stream_chunk_size}")
    
    # Initialize audio player with OutputStream
    player = None
    if play_audio:
        player = AudioPlayer(sample_rate=tts_model.sampling_rate)
        if player.backend:
            player.start()  # Start the continuous audio stream
    
    # Collect all audio chunks
    all_chunks = []
    chunk_count = 0
    start_time = time.time()
    first_chunk_time = None
    
    try:
        # Stream generation with incremental decoding
        for audio_chunk in tts_model.predict_stream(
            text=text,
            text_language=text_language,
            stream_chunk_size=stream_chunk_size,
        ):
            chunk_count += 1
            
            # Record first chunk latency
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
                logger.info(f"✨ First chunk generated in {first_chunk_time:.3f}s")
            
            # Write audio chunk to continuous stream (non-blocking)
            if player and player.backend and len(audio_chunk) > 0:
                player.write(audio_chunk)
            
            # Collect audio for saving
            all_chunks.append(audio_chunk)
            
            logger.debug(f"Chunk {chunk_count}: {len(audio_chunk)} samples, "
                        f"{len(audio_chunk)/tts_model.sampling_rate:.2f}s")
        
        # Wait for playback to finish
        if player and player.backend:
            # Wait for queue to be empty and pending to be consumed
            while not player.audio_queue.empty() or (player._pending.size > 0):
                time.sleep(0.05)
            time.sleep(0.2)  # Small extra time for last block
        
        total_time = time.time() - start_time
        logger.info(f"✅ Streaming completed: {chunk_count} chunks in {total_time:.3f}s")
        
        # Save complete audio if requested
        if save_path and all_chunks:
            import soundfile as sf
            # Concatenate all incremental chunks to get complete audio
            complete_audio = np.concatenate(all_chunks) if all_chunks else np.array([])
            sf.write(save_path, complete_audio, tts_model.sampling_rate)
            logger.info(f"💾 Complete audio saved to: {save_path}")
            logger.info(f"   Duration: {len(complete_audio)/tts_model.sampling_rate:.2f}s")
        
        return all_chunks
        
    finally:
        # Clean up
        if player:
            player.close()


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS Demo with Real-time Playback")
    parser.add_argument(
        "--speaker_model",
        type=str,
        default="shibing624/parrots-gpt-sovits-speaker-maimai",
        help="Speaker model path"
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="MaiMai",
        help="Speaker name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (FP16)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好，欢迎来到北京。这是一个流式语音合成的演示。Welcome to Beijing! This is a new demo.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="auto",
        help="Language: zh, en, ja, or auto"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="Number of semantic tokens per chunk (smaller = lower latency, but may affect quality)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_stream.wav",
        help="Path to save complete audio"
    )
    parser.add_argument(
        "--no_play",
        action="store_true",
        help="Disable real-time audio playback"
    )
    parser.add_argument(
        "--prepare_ref",
        action="store_true",
        help="Pre-prepare reference audio for lowest latency"
    )
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    # Initialize TTS model
    logger.info("Initializing TTS model...")
    tts = TextToSpeech(
        speaker_model_path=args.speaker_model,
        speaker_name=args.speaker_name,
        device=args.device,
        half=args.half
    )
    logger.info("✅ TTS model loaded successfully")
    
    # Pre-prepare reference audio if requested
    if args.prepare_ref:
        logger.info("Pre-preparing reference audio features...")
        tts.prepare_reference()
        logger.info("✅ Reference audio prepared and cached")
    
    # Run streaming TTS with playback
    logger.info("\n" + "="*60)
    logger.info("🎤 Starting Streaming TTS Generation")
    logger.info("="*60 + "\n")
    
    stream_tts_with_playback(
        tts_model=tts,
        text=args.text,
        text_language=args.lang,
        stream_chunk_size=args.chunk_size,
        save_path=args.output_path,
        play_audio=not args.no_play
    )
    
    logger.info("\n" + "="*60)
    logger.info("🎉 Streaming TTS Demo Completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
