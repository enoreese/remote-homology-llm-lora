import modal

image = modal.image.Image.debian_slim(  # we start from a lightweight linux distro
    python_version="3.10"  # we add a recent Python version
).pip_install(  # and we install the following packages:
    "google-cloud-storage",
    "requests",
    "python-dotenv",
    "tqdm",
    "pandas",
    "numpy",
    "jsonlines",
    "biopython",
    "scikit-learn"
)
volume = modal.volume.Volume.from_name("homology-volume", create_if_missing=True)

# we define a Stub to hold all the pieces of our app
# most of the rest of this file just adds features onto this Stub
stub = modal.App(
    name="homology_processor",
    image=image,
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
        modal.secret.Secret.from_name("my-googlecloud-secret")
    ]
)


def download_data(url, data_path='train_raw.json'):
    import requests

    print("Download raw data...")
    r = requests.get(url)
    with open(data_path, "w") as temp_data_file:
        print("Writing raw data...")
        temp_data_file.write(r.text)
    return True


@stub.function(volumes={'/vol': volume}, timeout=60 * 60 * 12, container_idle_timeout=1200)
def enter(download=False):
    import pandas as pd
    import os

    # cols = ['FA-DOMID', 'FA-PDBID', 'SF-DOMID', 'SF-PDBID', 'SF', 'FA', 'CF']
    base_url = "https://storage.googleapis.com/projects-bkt/llm-bkt/dna_files/scop/"

    volume.reload()
    print("Checking if path exists...")
    if not os.path.exists('/vol/scop/') or download:
        os.makedirs('/vol/scop/', exist_ok=True)
        download_data(f"{base_url}scop-cla-latest.csv", '/vol/scop/scop-cla-latest.csv')
        download_data(f"{base_url}scop_fa_represeq_lib_latest.csv", "/vol/scop/scop_fa_represeq_lib_latest.csv")
        download_data(f"{base_url}scop_sf_represeq_lib_latest.csv", "/vol/scop/scop_sf_represeq_lib_latest.csv")
        download_data(f"{base_url}scop-des-latest.csv", "/vol/scop/scop-des-latest.csv")

        volume.commit()

    if not os.path.exists('/vol/scop/results/'):
        os.makedirs('/vol/scop/results/', exist_ok=True)
        volume.commit()

    print("Loading DFs...")
    cla = pd.read_csv('/vol/scop/scop-cla-latest.csv')
    fa_seq = pd.read_csv('/vol/scop/scop_fa_represeq_lib_latest.csv', usecols=['DOMID', 'FA', 'sequence'])
    sf_seq = pd.read_csv('/vol/scop/scop_sf_represeq_lib_latest.csv', usecols=['DOMID', 'SF', 'sequence'])
    nodes_df = pd.read_csv('/vol/scop/scop-des-latest.csv')

    # cla = cla.drop_duplicates(subset=cols)
    cla = cla.reset_index()
    cla.rename({'index': 'uid'}, axis=1, inplace=True)

    process.remote(cla, fa_seq, sf_seq, nodes_df)


@stub.function(volumes={'/vol': volume}, timeout=60 * 60 * 12, retries=3, container_idle_timeout=1200)
def process_result(pairs_list, cla, fa_seq, sf_seq):
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()
    pairs_list, idx = pairs_list
    print(f"Len pairs list: {len(pairs_list)}, IDX: {idx}")

    # Convert your list of pairs into a DataFrame
    print("Dataframe...")
    pairs_df = pd.DataFrame(pairs_list, columns=['uid_a', 'uid_b'])
    print(pairs_df.shape)

    print("UID Merge...")
    results = pairs_df.merge(cla, left_on='uid_a', right_on='uid', how='left')
    results = results.merge(cla, left_on='uid_b', right_on='uid', how='left', suffixes=('_query', '_context'))
    results.drop(columns=['uid_a', 'uid_b'], inplace=True)
    print(results.shape)

    print("Calculating remote homologs...")
    results['remote_homologs'] = ((results['SF_query'] == results['SF_context']) & (
            results['FA_query'] != results['FA_context']))

    print("Getting FA sequence...")
    results['FA_seq_query'] = \
        pd.merge(results, fa_seq, how='left', left_on=['FA-DOMID_query', 'FA_query'], right_on=['DOMID', 'FA'])[
            'sequence']
    results['FA_seq_context'] = \
        pd.merge(results, fa_seq, how='left', left_on=['FA-DOMID_context', 'FA_context'], right_on=['DOMID', 'FA'])[
            'sequence']

    print("Getting SF sequence...")
    results['SF_seq_query'] = \
        pd.merge(results, sf_seq, how='left', left_on=['SF-DOMID_query', 'SF_query'], right_on=['DOMID', 'SF'])[
            'sequence']
    results['SF_seq_context'] = \
        pd.merge(results, sf_seq, how='left', left_on=['SF-DOMID_context', 'SF_context'], right_on=['DOMID', 'SF'])[
            'sequence']

    print(results.shape)
    print(results['remote_homologs'].value_counts(), end='\n')

    print("Saving to file...")
    filename = f'result_new_{str(idx)}.csv'
    results.to_csv(f'/vol/scop/results/{filename}')
    volume.commit()
    return True


# https://github.com/amoldwin/plm-zero-shot-remote-homology-evaluation/blob/main/preprocess_SCOP_data.ipynb
def filter_sequences(cla, nodes_df):
    import pandas as pd

    rossmann_fold_domains = nodes_df[nodes_df["NODE_NAME"].str.contains("Rossmann fold")]
    beta_propeller_4to8_domains = nodes_df[nodes_df["NODE_NAME"].str.contains(
        "4-bladed beta-propeller|5-bladed beta-propeller|6-bladed beta-propeller|7-bladed beta-propeller|8-bladed beta-propeller")]
    exclude_folds = pd.concat([rossmann_fold_domains, beta_propeller_4to8_domains])
    print("Removing datapoints classified into Rossmann folds or 4- to 8- beta-propellers...")
    filtered_scop_df = cla[~cla["CF"].isin(exclude_folds["NODE_ID"].to_list())]
    print("Removing datapoints where Family and Superfamily denoting sequence are different:")
    filtered_scop_df = filtered_scop_df[filtered_scop_df["FA-PDBREG"] == filtered_scop_df["SF-PDBREG"]]
    print("Removing datapoints where Family and Superfamily denoting PDB ids are different:")
    filtered_scop_df = filtered_scop_df[filtered_scop_df["FA-PDBID"] == filtered_scop_df["SF-PDBID"]]

    def have_one_span(x):
        return len(x.split(",")) == 1  # A:37-243,A:353-401

    print("Removing datapoints where Family denoting sequence have multiple spans:")
    filtered_scop_df = filtered_scop_df[filtered_scop_df["FA-PDBREG"].apply(have_one_span)]
    print("Removing datapoints where Superfamily denoting sequence have multiple spans:")
    filtered_scop_df = filtered_scop_df[filtered_scop_df["SF-PDBREG"].apply(have_one_span)]
    return filtered_scop_df


@stub.function(timeout=60 * 60 * 12, container_idle_timeout=1200)
def process(cla, fa_seq, sf_seq, nodes_df):
    import numpy as np
    from itertools import combinations

    print("Apply filtering...")
    cla = filter_sequences(cla, nodes_df)

    print("Generating combinations for: ", cla.shape[0])
    uids = cla.uid.values.tolist()
    pairs = list(combinations(uids, 2))
    print(len(uids), len(pairs), sep=' ', end='\n')

    print("Splitting...")
    splits = np.array_split(pairs, 100)

    print("Multi processing...")
    results = list(
        process_result.map(zip(splits, range(1, 100 + 1)), kwargs={'cla': cla, 'fa_seq': fa_seq, 'sf_seq': sf_seq}))


@stub.function(volumes={'/vol': volume}, timeout=60 * 60 * 12)
def merge_dfs():
    import os
    import pandas as pd
    from tqdm import tqdm

    volume.reload()

    print("Getting files...")
    files = [file for file in os.listdir('/vol/scop/results/') if
             file.endswith('.csv') and 'result_new_' in file]

    print(f"Looping {len(files)} files...")
    dfs = []
    for file in tqdm(files):
        filename = os.path.join('/vol/scop/results', file)
        df = pd.read_csv(filename, index_col=0)

        remote = df[df['remote_homologs'] == True]
        non_remote = df[df['remote_homologs'] == False].sample(remote.shape[0])

        df = pd.concat([remote, non_remote])
        df = df.sample(frac=1)

        print(df.shape)

        dfs.append(df)

    print(f"Concatenating {len(dfs)} dfs...")
    dfs = pd.concat(dfs)

    print(dfs.shape)
    print(dfs['remote_homologs'].value_counts(), end='\n')

    print("Saving to file...")
    dfs.to_csv('/vol/scop/results/combined_result_new.csv')
    volume.commit()


@stub.function(volumes={'/vol': volume}, timeout=60 * 60 * 12)
def calculate_identity():
    from Bio import pairwise2
    from tqdm import tqdm
    import os
    import json
    import pandas as pd
    from google.cloud import storage
    from google.oauth2 import service_account

    volume.reload()

    results = pd.read_csv('/vol/scop/results/combined_result_new.csv', index_col=0)

    print("Calculating identity...")
    results['FA_identity'] = results.apply(
        lambda row: pairwise2.align.globalxx(row['FA_seq_query'], row['FA_seq_context'], score_only=True), axis=1)
    # results['SF_identity'] = results.apply(
    #     lambda row: pairwise2.align.globalxx(row['SF_seq_query'], row['SF_seq_context'], score_only=True), axis=1)

    print("Calculating thresholds...")
    for threshold in tqdm(range(10, 100, 5)):
        print("Running threshold: ", threshold)
        results[f'remote_homologs{str(threshold)}'] = ((results['SF_query'] == results['SF_context']) & (
                results['FA_query'] != results['FA_context']) & (results['FA_identity'] <= threshold))

        print(results[f'remote_homologs{str(threshold)}'].value_counts())

    print("Saving to file...")
    results.to_csv('/vol/scop/results/combined_result_new_threshold.csv')
    volume.commit()

    print("Saving to GCP...")
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bucket = storage.Client(credentials=credentials).get_bucket(bucket_name)

    blob = bucket.blob(f'llm-bkt/dna_files/scop/combined_result_new.csv')
    blob.upload_from_filename(filename=f'/vol/scop/results/combined_result_new.csv', content_type='text/csv')

    blob = bucket.blob(f'llm-bkt/dna_files/scop/combined_result_new_threshold.csv')
    blob.upload_from_filename(filename=f'/vol/scop/results/combined_result_new_threshold.csv', content_type='text/csv')


def filename_to_gcp(blob_path, filepath, content_type='application/json'):
    import os
    import json
    from google.cloud import storage
    from google.oauth2 import service_account

    print("Saving to GCP...")
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bucket = storage.Client(credentials=credentials).get_bucket(bucket_name)

    blob = bucket.blob(blob_path)
    blob.upload_from_filename(filename=filepath, content_type=content_type)


def jsonlines_to_file(df_, filepath):
    import jsonlines
    from tqdm import tqdm

    with jsonlines.open(filepath, 'w') as w:
        for idx, row in tqdm(df_.iterrows()):
            record = {
                'fa_query': row['FA_seq_query'],
                'fa_context': row['FA_seq_context'],
                'sf_query': row['SF_seq_query'],
                'sf_context': row['SF_seq_context'],
                'output': row['remote_homologs']
            }
            w.write(record)


@stub.function(volumes={'/vol': volume}, timeout=60 * 60 * 12)
def create_dataset():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    columns = ["fa_query", 'fa_context', 'sf_query', 'sf_context', 'output']

    volume.reload()

    print("Loading df...")
    df = pd.read_csv('/vol/scop/results/combined_result_new_threshold.csv', index_col=0)
    df.rename({
        "FA_seq_query": "fa_query",
        'FA_seq_context': 'fa_context',
        'SF_seq_query': 'sf_query',
        'SF_seq_context': 'sf_context',
        'remote_homologs': 'output'
    }, axis='columns', inplace=True)

    print("Splitting dataset")
    train, test = train_test_split(df[columns], train_size=.8, stratify=df[['output']])
    print(train.shape, test.shape)

    print("Creating train...")
    train_path = "/vol/scop/results/soft_rh_train_new_dataset.json"
    train.to_json(train_path, orient='records', lines=True)
    filename_to_gcp("llm-bkt/dna_files/scop/soft_rh_train_new_dataset.json", train_path)

    print("Creating test...")
    test_path = "/vol/scop/results/soft_rh_test_new_dataset.json"
    test.to_json(test_path, orient='records', lines=True)
    filename_to_gcp("llm-bkt/dna_files/scop/soft_rh_test_new_dataset.json", test_path)

    volume.commit()
