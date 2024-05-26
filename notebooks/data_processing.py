import pandas as pd
from tqdm import tqdm

chunksize = 1000000

if __name__ == '__main__':
    data = []
    for chunk in tqdm(pd.read_csv('../datasets/scop/scop_pairs.csv', index_col=0, chunksize=chunksize)):
        homologs = chunk[chunk.remote_homologs == True]
        non_homologs = chunk[chunk.remote_homologs == False].sample(homologs.shape[0])

        print('Homologs: shape: ', homologs.shape)
        print('Non-Homologs: shape: ', non_homologs.shape)

        data_ = pd.concat([homologs, non_homologs], axis=0)
        print('Concat: shape: ', data_.shape)
        print(data_.columns)
        data.append(data_)

    data = pd.concat(data, axis=0)
    fa_seq = pd.read_csv('../datasets/scop/scop_fa_represeq_lib_latest.csv')
    sf_seq = pd.read_csv('../datasets/scop/scop_sf_represeq_lib_latest.csv')


    def merge_fn(row, seq=fa_seq, col='FA'):
        q_fa_mask = seq['DOMID'] == row[f'{col}-DOMID_query']
        q_fa_mask &= seq[col] == row[f'{col}_query']
        q_seq = seq[q_fa_mask].sequence.values[0]

        c_fa_mask = seq['DOMID'] == row[f'{col}-DOMID_context']
        c_fa_mask &= seq[col] == row[f'{col}_context']
        c_seq = seq[c_fa_mask].sequence.values[0]

        return q_seq, c_seq


    data[['FA_seq_query', 'FA_seq_context']] = data.apply(lambda row: merge_fn(row), axis=1, result_type='expand')
    data[['SF_seq_query', 'SF_seq_context']] = data.apply(lambda row: merge_fn(row, seq=sf_seq, col='SF'), axis=1,
                                                          result_type='expand')

    print(data.columns)
    print(data.shape)
    print(data.sample(1).values)

    data.to_csv('scop/scop_dataset.csv', index=False)
