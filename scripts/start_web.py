"""Start the portable web workspace; run from any working directory."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv

load_dotenv(ROOT / '.env', override=False)
os.environ.setdefault('CHAINLIT_APP_ROOT', str(ROOT / 'modes/ui'))
os.environ.setdefault('CHAINLIT_NO_TELEMETRY', 'true')
os.environ.setdefault('LANGCHAIN_TRACING_V2', 'false')
os.environ.setdefault('LANGSMITH_TRACING', 'false')
os.environ.setdefault('DO_NOT_TRACK', '1')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('modes.server:create_app', factory=True,
                host=os.environ.get('MDRPI_MODES_HOST', '127.0.0.1'),
                port=int(os.environ.get('MDRPI_MODES_PORT', '8021')), workers=1)
