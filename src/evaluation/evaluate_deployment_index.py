from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src' / 'retrieval') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src' / 'retrieval'))
from deployment_rag import DeploymentRetriever, deduplicate_retrieved  # noqa: E402

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as exc: raise ValueError(f'Invalid JSONL at {path}:{line_no}: {exc}') from exc
    return rows

def mean_or_none(values: list[float]) -> float | None:
    return sum(values)/len(values) if values else None

def metrics(ranked: list[str], relevant: list[str], k: int) -> dict[str, float | None]:
    if not relevant: return {f'hit@{k}':None, f'recall@{k}':None, f'mrr@{k}':None, f'ndcg@{k}':None}
    rel=set(relevant); top=ranked[:k]; positions=[i for i,cid in enumerate(top,1) if cid in rel]
    dcg=sum(1/math.log2(rank+1) for rank in positions); idcg=sum(1/math.log2(rank+1) for rank in range(1,min(len(rel),k)+1))
    return {f'hit@{k}':1.0 if positions else 0.0, f'recall@{k}':len(set(top)&rel)/len(rel), f'mrr@{k}':1/positions[0] if positions else 0.0, f'ndcg@{k}':dcg/idcg if idcg else 0.0}

def main() -> int:
    parser=argparse.ArgumentParser(description='Evaluate FastEmbed deployment retrieval against the labeled seed set.')
    parser.add_argument('--eval-set',default='data/evaluation/eval_set.jsonl'); parser.add_argument('--top-k',type=int,default=8); parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.top_k<1: parser.error('--top-k must be >= 1')
    eval_path=(ROOT/args.eval_set).resolve()
    if not eval_path.exists(): raise SystemExit(f'Evaluation set not found: {eval_path}')
    cases=read_jsonl(eval_path)
    if args.limit>0: cases=cases[:args.limit]
    retriever=DeploymentRetriever.from_directory()
    print(f'Evaluation cases: {len(cases)}'); print(f'Top K: {args.top_k}'); print(f'Embedding model: {retriever.embedder.__class__.__name__} (matching index manifest)')
    rows=[]; scored=[]; by_category: dict[str,list[dict[str,float]]]=defaultdict(list)
    for i,case in enumerate(cases,1):
        raw=retriever.search(case['question'],top_k=args.top_k,paper_id=case.get('paper_id')); deduped,removed=deduplicate_retrieved(raw); results=deduped[:args.top_k]
        ids=[str(r.get('chunk_id','')) for r in results]; relevant=[str(x) for x in case.get('relevant_chunk_ids',[])]; answerable=bool(case.get('answerable',True))
        score=metrics(ids,relevant,args.top_k) if answerable else metrics(ids,[],args.top_k)
        row={'id':case.get('id',f'case_{i}'),'category':case.get('category','uncategorized'),'paper_id':case.get('paper_id'),'answerable':answerable,'label_status':case.get('label_status','unspecified'),'relevant_chunk_ids':relevant,'retrieved_chunk_ids':ids,'metrics':score,'dedup_removed':[x.get('removed_chunk_id') for x in removed],'top_sources':[{'chunk_id':r.get('chunk_id'),'source_type':r.get('source_type'),'paper_id':r.get('paper_id'),'section':r.get('section'),'page_number':r.get('page_number')} for r in results]}
        rows.append(row)
        if answerable: scored.append(score); by_category[row['category']].append(score)
        hit=score.get(f'hit@{args.top_k}'); print(f"[{i}/{len(cases)}] {row['id']}: "+(f'Hit@{args.top_k}={hit:.0%}' if hit is not None else 'unanswerable; ranking metrics excluded'))
    names=[f'hit@{args.top_k}',f'recall@{args.top_k}',f'mrr@{args.top_k}',f'ndcg@{args.top_k}']
    overall={n:mean_or_none([float(s[n]) for s in scored if s.get(n) is not None]) for n in names}
    categories={cat:{n:mean_or_none([float(s[n]) for s in vals if s.get(n) is not None]) for n in names} for cat,vals in by_category.items()}
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'); out=ROOT/'data'/'evaluation'/'deployment_runs'/timestamp; out.mkdir(parents=True,exist_ok=True)
    warning='Seed labels are drafts and must be manually verified. This measures retrieval only, not answer correctness, citation entailment, multimodal understanding, abstention quality, or plagiarism.'
    report={'timestamp':timestamp,'eval_set':str(eval_path.relative_to(ROOT)),'case_count':len(cases),'answerable_cases':len(scored),'unanswerable_cases_excluded_from_ranking_metrics':sum(not bool(c.get('answerable',True)) for c in cases),'top_k':args.top_k,'overall':overall,'by_category':categories,'warning':warning,'per_case':rows}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Deployment Retrieval Evaluation','',f"- Cases: {len(cases)}",f'- Top K: {args.top_k}',f'- Answerable cases scored: {len(scored)}','','## Overall metrics','','| Metric | Score |','|---|---:|']+[f"| {n} | {'N/A' if v is None else f'{v:.4f}'} |" for n,v in overall.items()]
    lines+=['','## Per-category metrics','','| Category | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    for cat,vals in categories.items(): lines.append('| '+cat+' | '+' | '.join('N/A' if vals[n] is None else f'{vals[n]:.4f}' for n in names)+' |')
    lines+=['','## Caveat','',warning,'','## Per-case results','']
    for r in rows: lines += [f"### {r['id']}",f"- Scope: `{r['paper_id']}`",f"- Retrieved: `{', '.join(r['retrieved_chunk_ids'])}`",f"- Relevant labels: `{', '.join(r['relevant_chunk_ids'])}`",'']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\nOverall retrieval metrics:')
    for n,v in overall.items(): print(f"  {n}: {'N/A' if v is None else f'{v:.4f}'}")
    print(f'Saved report: {out / "report.md"}')
    return 0
if __name__=='__main__': raise SystemExit(main())
