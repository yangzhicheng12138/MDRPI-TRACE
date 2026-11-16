"""Check the public Git snapshot without printing any matching secret values."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.pytest_cache', '.venv', 'node_modules'}
FORBIDDEN = {'.sqlite', '.db', '.safetensors', '.gguf', '.pdf', '.docx', '.pem', '.key'}
PREFIXES = ('logs/', 'reports/', '.secrets/', 'data/interactive_modes/', 'data/private_research/',
            'data/models/', 'data/source/', 'data/qdrant/', 'modes/ui/.files/')
PATTERNS = {
    'credential': re.compile(r'(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)'),
    'private deployment path': re.compile(r'/(?:hd_4t|lxm_4t)/[^\s]+|/(?:home|Users)/[a-z0-9_.-]+/|\b10(?:\.\d{1,3}){3}\b'),
}
# Non-working fixtures, accepted only in the tests that use them.
SENTINELS = {
    'sk-' + 'controlled1234567890': {'tests/test_search_planner.py'},
    '10.' + '0.0.2': {'tests/test_reflection_v2.py'},
}


def files():
    try:
        top = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'], stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top) == ROOT:
            names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'], text=True).split('\0')
            if any(names):
                return [ROOT / n for n in names if n]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p for p in ROOT.rglob('*') if p.is_file() and not any(s in SKIP for s in p.relative_to(ROOT).parts)
            and not p.relative_to(ROOT).as_posix().startswith(('data/interactive_modes/', 'logs/'))]


def main():
    problems = []
    checked = files()
    for p in checked:
        name = p.relative_to(ROOT).as_posix()
        if (p.is_symlink() or name.startswith(PREFIXES) or p.suffix in FORBIDDEN
                or '.sqlite' in p.name or p.name == '.env' or p.stat().st_size > 10_000_000):
            problems.append(f'{name}: private/runtime or oversized file')
        try:
            text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                matches = [m.group() for m in pattern.finditer(line)]
                if any(name not in SENTINELS.get(m, set()) for m in matches):
                    problems.append(f'{name}:{number}: {kind}')
    for name in ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'config.yaml', '.env.example']:
        if not (ROOT / name).is_file():
            problems.append(f'{name}: required file missing')
    if problems:
        print('\n'.join(problems))
        return 1
    print(f'Public release check passed: {len(checked)} files; no detected private assets or credentials.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
