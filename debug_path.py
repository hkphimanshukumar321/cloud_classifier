import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
try:
    from classification import config
    print(f"Config file location: {config.__file__}")
    print(f"Resolved data_dir: {config.ResearchConfig().data.data_dir}")
except Exception as e:
    print(f"Error: {e}")
