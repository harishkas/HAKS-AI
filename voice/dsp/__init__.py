"""HAKS Digital Signal Processing (DSP) and Human Voice Isolation Subsystem."""

from voice.dsp.dc_normalizer import DCNormalizer
from voice.dsp.echo_canceller import AcousticEchoCanceller, AECEvaluationResult
from voice.dsp.noise_suppressor import NoiseSuppressor, NoiseSuppressionStrength
from voice.dsp.speech_enhancer import SpeechEnhancer
from voice.dsp.vad_detector import AdaptiveVoiceDetector, VADFrameResult
from voice.dsp.pipeline import AudioProcessingPipeline, ProcessedAudioFrame

__all__ = [
    "DCNormalizer",
    "AcousticEchoCanceller",
    "AECEvaluationResult",
    "NoiseSuppressor",
    "NoiseSuppressionStrength",
    "SpeechEnhancer",
    "AdaptiveVoiceDetector",
    "VADFrameResult",
    "AudioProcessingPipeline",
    "ProcessedAudioFrame",
]

