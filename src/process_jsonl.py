from kmer import kmer_featurization
import json, jsonlines

k = 6
featurizer = kmer_featurization(5)

with open("../datasets/my_data_fixed.jsonl", 'r') as r, jsonlines.open('../datasets/my_data_fixed_2.jsonl', 'w')as w:
  for line in r:
    record = json.loads(line)
    
    seq = record['context'].split(" ")
    kmer_feature = featurizer.obtain_kmer_feature_for_a_list_of_sequences(seq)
    
    record['question'] = record['question'].strip()
    record['context'] = kmer_feature
    record['answer'] = record['answer'].strip()
    w.write(record)
    # break