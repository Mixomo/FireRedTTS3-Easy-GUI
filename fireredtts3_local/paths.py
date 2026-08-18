from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
VOICES_DIR = ROOT / "voices"
SAMPLES_DIR = ROOT / "samples"
CACHE_DIR = MODELS_DIR / ".cache"
MODEL_DIR = MODELS_DIR / "FireRedTTS3"
SETTINGS_DIR = ROOT / "config"
SETTINGS_FILE = SETTINGS_DIR / "ui_settings.json"
VOICE_INDEX = VOICES_DIR / "voices.json"
CPP_ROOT = ROOT / ".runtime" / "firered-cpp"
CPP_BUNDLES_DIR = ROOT / "models" / "fireredtts3-cpp"


def ensure_dirs():
    for path in (MODELS_DIR, OUTPUTS_DIR, VOICES_DIR, SAMPLES_DIR, CACHE_DIR, SETTINGS_DIR, CPP_ROOT, CPP_BUNDLES_DIR):
        path.mkdir(parents=True, exist_ok=True)
