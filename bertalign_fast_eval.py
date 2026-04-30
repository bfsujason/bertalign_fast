# bfsujason@163.com

"""
Usage examples:
    python bertalign_fast_eval.py --dataset berg
    python bertalign_fast_eval.py --dataset mac
    
"""

import os
import time
import argparse

from bertalign_fast import BertalignFast
from bertalign_fast.eval import (
    read_alignments,
    score_multiple,
    log_final_scores,
)

def main():
    parser = argparse.ArgumentParser(description="Parallel corpus aligment using Bertalign-Fast.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--split", action="store_true", help="Split sentences.")
    args = parser.parse_args()
    
    aligner = BertalignFast()
    #print(f"Supported languages: {aligner.languages}\n")
    
    src_dir = f"data/{args.dataset}/src"
    tgt_dir = f"data/{args.dataset}/tgt"
    gold_dir = f"data/{args.dataset}/gold"

    test_alignments = []
    gold_alignments = []
    for file in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file).replace("\\","/")
        tgt_file = os.path.join(tgt_dir, file).replace("\\","/")
        src = open(src_file, 'rt', encoding='utf-8').read()
        tgt = open(tgt_file, 'rt', encoding='utf-8').read()

        print(f"Start aligning {src_file} to {tgt_file}")
        aligner.align_sents(src, tgt, split=args.split)
        #print(aligner.result)

        test_alignments.append([(x, y) for x, y, score in aligner.result])

        gold_file = os.path.join(gold_dir, file)
        gold_alignments.append(read_alignments(gold_file))
        
    scores = score_multiple(gold_list=gold_alignments, test_list=test_alignments)
    log_final_scores(scores)

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total time spent: {elapsed:.2f} secs\n")
