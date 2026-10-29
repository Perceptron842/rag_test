"""Score only HUMAN-LABELED correctness/faithfulness columns (0/1)."""
import argparse,csv,json
from pathlib import Path

def evaluate_rows(rows):
    valid=[]
    for row in rows:
        a,b=str(row.get('correctness','')).strip(),str(row.get('faithfulness','')).strip()
        if not a and not b: continue
        if a not in ('0','1') or b not in ('0','1'):
            raise ValueError('Both labels must be 0 or 1 once a row is reviewed')
        valid.append((int(a),int(b)))
    if not valid: return {'reviewed':0,'message':'No manually labeled rows yet'}
    return {'reviewed':len(valid), 'answer_correctness_rate':sum(a for a,b in valid)/len(valid),
            'faithfulness_rate':sum(b for a,b in valid)/len(valid)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='outputs/answer_review.csv');args=p.parse_args()
    with Path(args.input).open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    print(json.dumps(evaluate_rows(rows),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
