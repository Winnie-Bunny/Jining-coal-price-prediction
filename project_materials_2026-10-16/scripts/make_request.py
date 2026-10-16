"""Generate a complete inference request from the original CSV, using stdlib only."""
import argparse
import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATERIALS_ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    '京唐港瘦煤(加)（元/吨）': '京唐港瘦煤（加）（元/吨）',
    '龙煤鹤岗1/3焦煤(A8.5)（元/吨）': '龙煤集团鹤岗1/3焦煤(A8.5）（元/吨）',
    '龙煤鹤岗1/3焦煤(A9.5)（元/吨）': '龙煤集团鹤岗1/3焦煤（A9.5）（元/吨）',
    '上海港主焦煤(澳)（元/吨）': '上海港主焦煤（澳）（元/吨）',
    '天津港主焦煤(蒙古)（元/吨）': '天津港主焦煤（蒙古）（元/吨）',
    '天峻焦煤(G70)（元/吨）': '天峻焦煤（G70）（元/吨）',
    '天峻焦煤(G80)（元/吨）': '天峻焦煤（G80）（元/吨）',
    '乌海肥精煤(G>88)（元/吨）': '乌海肥精煤（G>88)（元/吨）',
    '乌海肥精煤(G90)（元/吨）': '乌海肥精煤（G90）（元/吨）',
    '冶金煤:内蒙古策克1/3焦煤（元/吨）': '冶金煤:内蒙古策克蒙古1/3焦煤（元/吨）',
    '日照港主焦煤(澳)（元/吨）': '日照港主焦煤（澳）（元/吨）',
}


def make_request(model_id):
    mapping = json.loads((MATERIALS_ROOT / 'docs/feature-map.json').read_text())
    with (ROOT / 'data/cci_jining_coal.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    # Drop exact duplicate records only; do not choose silently between conflicting values.
    by_date = {}
    for row in rows:
        if row['date'] in by_date and by_date[row['date']] != row:
            raise ValueError(f"Conflicting records for {row['date']}")
        by_date[row['date']] = row
    rows = sorted(by_date.values(), key=lambda row: datetime.datetime.strptime(row['date'], '%Y/%m/%d'))
    count = {7: 60, 30: 90, 365: 364}[model_id]
    if len(rows) < count:
        raise ValueError('Not enough historical records')
    records = []
    for row in rows[-count:]:
        record = {'date': datetime.datetime.strptime(row['date'], '%Y/%m/%d').date().isoformat(),
                  'history_price': float(row['历史价格'])}
        record.update({key: float(row[ALIASES.get(name, name)]) for key, name in mapping.items()})
        records.append(record)
    return {'id': model_id, 'price_data': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', type=int, choices=(7, 30, 365), default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(make_request(args.id), ensure_ascii=False, indent=2) + '\n')
