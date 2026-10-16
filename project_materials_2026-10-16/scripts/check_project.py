"""Check Python syntax, feature consistency and required model assets without loading pickle."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATERIALS_ROOT = Path(__file__).resolve().parents[1]


def check():
    mapping = json.loads((MATERIALS_ROOT / 'docs/feature-map.json').read_text())
    for source in ROOT.rglob('*.py'):
        if any(part.startswith('.') for part in source.relative_to(ROOT).parts):
            continue
        ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    for period in ('week2week', 'month2month', 'year2year'):
        folder = ROOT / 'inference' / period
        tree = ast.parse((folder / 'predict_price.py').read_text())
        actual = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'FEATURE_MAP' for t in n.targets))
        assert list(actual.items()) == list(mapping.items()), f'{period}: feature order differs'
        required = [folder / 'model' / name for name in
                    ('SVR_BG.pkl', 'input_StandardScaler.pkl', 'price_StandardScaler.pkl')]
        for idx in range(len(mapping)):
            required.extend((folder / 'lstm_models' / f'lstm_{idx}.pt',
                             folder / 'model' / f'{idx}_input_x_StandardScaler.pkl',
                             folder / 'model' / f'{idx}_input_y_StandardScaler.pkl'))
        for path in required:
            assert path.is_file() and path.stat().st_size > 0, f'Missing/empty: {path}'
        print(f'{period}: {len(mapping)} LSTMs and all required scalers present')
    for line in (MATERIALS_ROOT / 'docs/artifacts.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print('Python syntax, feature maps, model assets and SHA-256 checks passed')


if __name__ == '__main__':
    check()
