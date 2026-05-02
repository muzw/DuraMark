"""DuraMark: Duration-based audio watermarking for speech synthesis."""

import os
import sys
import warnings
import logging

from duramark._version import __version__

# Suppress noisy third-party logging and warnings at import time
warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
try:
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
except ImportError:
    pass
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Inject Matcha-TTS third-party dependency path
_duramark_dir = os.path.dirname(os.path.abspath(__file__))
_matcha_path = os.path.join(_duramark_dir, '..', 'third_party', 'Matcha-TTS')
if os.path.isdir(_matcha_path) and _matcha_path not in sys.path:
    sys.path.insert(0, _matcha_path)


def _setup_cosyvoice_aliases():
    """Create fully traversable cosyvoice module aliases for YAML configs.

    HyperPyYAML uses pydoc.locate() to resolve class paths like
    'cosyvoice.llm.llm.TransformerLM'. pydoc.locate() walks the module
    hierarchy step by step and expects each submodule to be an attribute
    of its parent. We alias cosyvoice.* -> duramark.tts.* and also link
    duramark.detector submodules under cosyvoice.detector.*
    """
    import importlib
    import duramark.tts
    import duramark.tts.cosyvoice
    import duramark.tts.model
    import duramark.tts.frontend
    import duramark.tts.llm.llm
    import duramark.tts.flow.flow
    import duramark.tts.flow.decoder
    import duramark.tts.flow.flow_matching
    import duramark.tts.flow.length_regulator
    import duramark.tts.hifigan.generator
    import duramark.tts.hifigan.discriminator
    import duramark.tts.hifigan.hifigan
    import duramark.tts.hifigan.f0_predictor
    import duramark.tts.transformer.encoder
    import duramark.tts.transformer.encoder_layer
    import duramark.tts.transformer.decoder
    import duramark.tts.transformer.decoder_layer
    import duramark.tts.transformer.attention
    import duramark.tts.transformer.embedding
    import duramark.tts.transformer.convolution
    import duramark.tts.transformer.subsampling
    import duramark.tts.transformer.positionwise_feed_forward
    import duramark.tts.transformer.activation
    import duramark.tts.transformer.label_smoothing_loss
    import duramark.tts.transformer.upsample_encoder
    import duramark.tts.tokenizer.tokenizer
    import duramark.tts.utils.common
    import duramark.tts.utils.mask
    import duramark.tts.utils.file_utils
    import duramark.tts.utils.frontend_utils
    import duramark.tts.utils.class_utils
    import duramark.detector.model
    import duramark.detector.aligner
    import duramark.detector.downsampler
    import duramark.dataset as _dataset_stub

    # Build the cosyvoice namespace module
    cosyvoice = duramark.tts
    sys.modules['cosyvoice'] = cosyvoice

    # cosyvoice.llm
    cosyvoice.llm = duramark.tts.llm
    sys.modules['cosyvoice.llm'] = duramark.tts.llm
    cosyvoice.llm.llm = duramark.tts.llm.llm
    sys.modules['cosyvoice.llm.llm'] = duramark.tts.llm.llm

    # cosyvoice.flow
    cosyvoice.flow = duramark.tts.flow
    sys.modules['cosyvoice.flow'] = duramark.tts.flow
    cosyvoice.flow.flow = duramark.tts.flow.flow
    sys.modules['cosyvoice.flow.flow'] = duramark.tts.flow.flow
    cosyvoice.flow.decoder = duramark.tts.flow.decoder
    sys.modules['cosyvoice.flow.decoder'] = duramark.tts.flow.decoder
    cosyvoice.flow.flow_matching = duramark.tts.flow.flow_matching
    sys.modules['cosyvoice.flow.flow_matching'] = duramark.tts.flow.flow_matching
    cosyvoice.flow.length_regulator = duramark.tts.flow.length_regulator
    sys.modules['cosyvoice.flow.length_regulator'] = duramark.tts.flow.length_regulator

    # cosyvoice.hifigan
    cosyvoice.hifigan = duramark.tts.hifigan
    sys.modules['cosyvoice.hifigan'] = duramark.tts.hifigan
    cosyvoice.hifigan.generator = duramark.tts.hifigan.generator
    sys.modules['cosyvoice.hifigan.generator'] = duramark.tts.hifigan.generator
    cosyvoice.hifigan.discriminator = duramark.tts.hifigan.discriminator
    sys.modules['cosyvoice.hifigan.discriminator'] = duramark.tts.hifigan.discriminator
    cosyvoice.hifigan.hifigan = duramark.tts.hifigan.hifigan
    sys.modules['cosyvoice.hifigan.hifigan'] = duramark.tts.hifigan.hifigan
    cosyvoice.hifigan.f0_predictor = duramark.tts.hifigan.f0_predictor
    sys.modules['cosyvoice.hifigan.f0_predictor'] = duramark.tts.hifigan.f0_predictor

    # cosyvoice.transformer
    cosyvoice.transformer = duramark.tts.transformer
    sys.modules['cosyvoice.transformer'] = duramark.tts.transformer
    for _name in [
        'encoder', 'encoder_layer', 'decoder', 'decoder_layer',
        'attention', 'embedding', 'convolution', 'subsampling',
        'positionwise_feed_forward', 'activation', 'label_smoothing_loss',
        'upsample_encoder',
    ]:
        _mod = getattr(duramark.tts.transformer, _name)
        setattr(cosyvoice.transformer, _name, _mod)
        sys.modules[f'cosyvoice.transformer.{_name}'] = _mod

    # cosyvoice.tokenizer
    cosyvoice.tokenizer = duramark.tts.tokenizer
    sys.modules['cosyvoice.tokenizer'] = duramark.tts.tokenizer
    cosyvoice.tokenizer.tokenizer = duramark.tts.tokenizer.tokenizer
    sys.modules['cosyvoice.tokenizer.tokenizer'] = duramark.tts.tokenizer.tokenizer

    # cosyvoice.utils
    cosyvoice.utils = duramark.tts.utils
    sys.modules['cosyvoice.utils'] = duramark.tts.utils
    for _name in ['common', 'mask', 'file_utils', 'frontend_utils', 'class_utils']:
        _mod = getattr(duramark.tts.utils, _name)
        setattr(cosyvoice.utils, _name, _mod)
        sys.modules[f'cosyvoice.utils.{_name}'] = _mod

    # cosyvoice.cli aliases
    cosyvoice.cli = duramark.tts
    sys.modules['cosyvoice.cli'] = duramark.tts
    sys.modules['cosyvoice.cli.cosyvoice'] = duramark.tts.cosyvoice
    sys.modules['cosyvoice.cli.model'] = duramark.tts.model
    sys.modules['cosyvoice.cli.frontend'] = duramark.tts.frontend

    # cosyvoice.detector -> duramark.detector
    cosyvoice.detector = duramark.detector
    sys.modules['cosyvoice.detector'] = duramark.detector
    cosyvoice.detector.duration_detector = duramark.detector.model
    sys.modules['cosyvoice.detector.duration_detector'] = duramark.detector.model
    cosyvoice.detector.aligner = duramark.detector.aligner
    sys.modules['cosyvoice.detector.aligner'] = duramark.detector.aligner
    cosyvoice.detector.downsampler = duramark.detector.downsampler
    sys.modules['cosyvoice.detector.downsampler'] = duramark.detector.downsampler

    # cosyvoice.dataset.processor -> stub (training-only, never called during inference)
    cosyvoice.dataset = _dataset_stub
    cosyvoice.dataset.processor = _dataset_stub
    sys.modules['cosyvoice.dataset'] = _dataset_stub
    sys.modules['cosyvoice.dataset.processor'] = _dataset_stub


_setup_cosyvoice_aliases()
del _setup_cosyvoice_aliases

__all__ = ["__version__"]
