import json
from pathlib import Path
from collections import Counter, defaultdict

base = Path('memory/long_term')
files = list(base.iterdir()) if base.exists() else []
print('BASE', base.resolve())
print('total files', len(files))

cat = Counter()
for p in files:
    n = p.name
    if n == 'index.json':
        cat['index.json'] += 1
    elif n == 'index.pending.json':
        cat['index.pending.json'] += 1
    elif n.startswith('index.') and n.endswith('.tmp'):
        cat['index.*.tmp'] += 1
    elif '.corrupt_' in n and n.endswith('.bak'):
        cat['corrupt bak'] += 1
    elif n.startswith('memory_item_') and p.suffix == '.json':
        cat['memory_item json'] += 1
    elif n.startswith('summary_') and p.suffix == '.json':
        cat['summary json'] += 1
    else:
        cat['other'] += 1
print('categories', dict(cat))

print('non canonical files:')
for p in files:
    canonical = p.name == 'index.json' or (p.name.startswith('memory_item_') and p.suffix == '.json') or (p.name.startswith('summary_') and p.suffix == '.json')
    if not canonical:
        print(' ', p.name, p.stat().st_size)

items = []
summaries = []
bad = []
for p in base.glob('memory_item_*.json'):
    try:
        items.append((p, json.loads(p.read_text(encoding='utf-8'))))
    except Exception as e:
        bad.append((p, str(e)))
for p in base.glob('summary_*.json'):
    try:
        summaries.append((p, json.loads(p.read_text(encoding='utf-8'))))
    except Exception as e:
        bad.append((p, str(e)))
print('parse bad', len(bad))
for p, e in bad[:20]:
    print('BAD', p.name, e)
print('items', len(items), 'summaries', len(summaries))
print('kind', dict(Counter(str(d.get('kind') or d.get('type') or '') for _, d in items)))
print('status', dict(Counter(str(d.get('status') or '') for _, d in items)))

missing = []
for p, d in items:
    req = ['id', 'title', 'content', 'created_at', 'updated_at', 'importance', 'confidence', 'status']
    miss = [k for k in req if k not in d or d.get(k) in (None, '')]
    if miss:
        missing.append((p.name, miss, str(d.get('title', ''))[:60]))
print('missing required-ish', len(missing))
for row in missing[:50]:
    print('MISSING', row)

ids = defaultdict(list)
fp = defaultdict(list)
for p, d in items:
    ids[d.get('id')].append(p.name)
    key = (str(d.get('kind') or d.get('type') or ''), str(d.get('title', '')).strip(), str(d.get('content', '')).strip())
    fp[key].append((p.name, d.get('updated_at', ''), d.get('status', ''), d.get('importance')))
print('duplicate ids', sum(1 for v in ids.values() if len(v) > 1))
for k, v in ids.items():
    if len(v) > 1:
        print('DUPID', k, v)
print('duplicate exact content groups', sum(1 for v in fp.values() if len(v) > 1))
shown=0
for k, v in fp.items():
    if len(v) > 1:
        print('DUPCONTENT', k[0], k[1][:100], 'count', len(v))
        for x in v[:8]:
            print(' ', x)
        shown += 1
        if shown >= 30:
            break

low = []
for p, d in items:
    title = str(d.get('title', '')).strip()
    content = str(d.get('content', '')).strip()
    if (len(title) < 5 or len(content) < 20 or title.startswith('工具执行') or '工具执行成功' in title or '工具执行失败' in title or content.startswith('STDOUT:') or content.startswith('[SEARCH]') or content.startswith('--- 文件:')):
        low.append((p.name, d.get('kind') or d.get('type'), title[:100], len(content), d.get('importance'), d.get('status')))
print('low-quality heuristic', len(low))
for row in low[:150]:
    print('LOW', row)

idx = base / 'index.json'
if idx.exists():
    data = json.loads(idx.read_text(encoding='utf-8'))
    print('index top counts', {k: len(v) if hasattr(v, '__len__') else None for k, v in data.items()})
    missing_paths = []
    for id_, path in data.get('item_index', {}).items():
        if not Path(path).exists():
            missing_paths.append((id_, path))
    print('index missing item paths', len(missing_paths))
