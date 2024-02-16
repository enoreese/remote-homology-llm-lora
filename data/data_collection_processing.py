from retrying import retry
from google.cloud import storage
from google.oauth2 import service_account
import requests
import json
import os
import re
import modal
from tqdm import tqdm
from dotenv import load_dotenv
from requests.exceptions import RequestException
from requests.exceptions import ChunkedEncodingError


load_dotenv()

FILE_PATH = "/dna_files/dna_ids_file.txt"
DATA_URL = "https://storage.googleapis.com/projects-bkt/llm-bkt/dna_files/dna_ids_file.txt"

# definition of our container image for jobs on Modal
# Modal gets really powerful when you start using multiple images!
image = modal.image.Image.debian_slim(  # we start from a lightweight linux distro
    python_version="3.10"  # we add a recent Python version
).pip_install(  # and we install the following packages:
    "google-cloud-storage",
    "requests",
    "python-dotenv",
    "tqdm",
    "retrying"
)
volume = modal.volume.Volume.persisted("dna-volume")


# we define a Stub to hold all the pieces of our app
# most of the rest of this file just adds features onto this Stub
stub = modal.stub.Stub(
    name="dna-data-scraper",
    image=image,
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
        modal.secret.Secret.from_name("my-googlecloud-secret")
    ]
)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# @retry(stop_max_attempt_number=3, wait_fixed=5000)  # Retry 3 times with a 5-second delay between retries
def search_pubdb(database, query, ret_type='text'):
    try:
        url = BASE_URL + f"esearch.fcgi?db={database}&rettype={ret_type}&term={query}&usehistory=y"
        
        response = requests.get(url)
        response.raise_for_status()
        return response
    except (RequestException, ChunkedEncodingError) as e:
        # Handle the specific error (ChunkedEncodingError) here
        print(f"Error fetching page: {e}")
        return None
    

def get_ids_list(database, query, batch_size=500):
    res = search_pubdb(database, query)
    if res is None:
        return None
    
    if res.status_code != 200:
        return None
        
    xm_str = res.text
    web_env_match = re.search(r'<WebEnv>(\S+)<\/WebEnv>', xm_str)
    query_key_match = re.search(r'<QueryKey>(\d+)<\/QueryKey>', xm_str)
    count_match = re.search(r'<Count>(\d+)<\/Count>', xm_str)

    web = web_env_match.group(1) if web_env_match else None
    key = query_key_match.group(1) if query_key_match else None
    count = count_match.group(1) if count_match else None

    # Print or use the extracted values
    print("WebEnv: ", web)
    print("QueryKey: ", key)
    print("Count: ", count)

    # Loop through batches
    data = []
    params_list = [(r, web, key, batch_size) for r in range(0, int(count), batch_size)]
    print(f"Collecting {len(params_list)} batches...")
    for result in fetch.map(params_list):
        data.append(result)

    print("Writing to file...")
    with open(FILE_PATH, "w") as out_file:
        for efetch_out in tqdm(data):
            if efetch_out is not None:
                out_file.write(efetch_out)
            
    print("Saving to GCP...")
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bucket = storage.Client(credentials=credentials).get_bucket(bucket_name)

    blob = bucket.blob('llm-bkt/dna_files/dna_ids_file.txt')
    # take the upload outside of the for-loop otherwise you keep overwriting the whole file
    blob.upload_from_file(open(FILE_PATH, 'r'), content_type='text/plain')
    return True


@stub.function(concurrency_limit=500, timeout=86400)
def fetch(params):
    response = efetch(params=params)
    return response


@retry(stop_max_attempt_number=3, wait_fixed=5000)
def efetch(params):
    try:
        efetch_url = f"{BASE_URL}efetch.fcgi?db=nucleotide&WebEnv={params[1]}"
        efetch_url += f"&query_key={params[2]}&retstart={params[0]}"
        efetch_url += f"&retmax={params[3]}&rettype=genbankfull&retmode=text"

        # Make the request
        efetch_out = requests.get(efetch_url).text
        return efetch_out
    except (RequestException, ChunkedEncodingError) as e:
        # Handle the specific error (ChunkedEncodingError) here
        print(f"Error fetching page: {e}")
        return None


@stub.function(concurrency_limit=500, timeout=86400)
def parse_lines(lines):
  lines = lines.splitlines()
  result_dict = {}

  current_key = None
  current_value = ''
  reference_counter = 0

  for line in lines:
    match = re.match(r'^(\S+)\s*(.*)$', line)
    if match:
      if current_key is not None:
        if (current_key == 'REFERENCE') and (current_key in result_dict):
            current_key = "REFERENCE" + f"_{reference_counter}"
            reference_counter += 1
        
        result_dict[current_key] = current_value.strip()
      current_key, current_value = match.groups()
    else:
      current_value += ' ' + line

    if current_key is not None:
      if (current_key == 'REFERENCE') and (current_key in result_dict):
        current_key = "REFERENCE" + f"_{reference_counter}"
        reference_counter += 1
      result_dict[current_key] = current_value.strip()

  return result_dict


def parse_data():
    data = []
    print("Getting data...")
    response = requests.get(DATA_URL)
    text_file =  response.text

    print("Splitting data...")
    groups = text_file.split('//')

    try:
        print(f"Parsing {len(groups)} groups...")
        for result in parse_lines.map(groups):
            data.append(result)

    except KeyboardInterrupt:
        pass

    print("Saving to GCP...")
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bucket = storage.Client(credentials=credentials).get_bucket(bucket_name)

    blob = bucket.blob('llm-bkt/dna_files/dna_corpus.json')
    # take the upload outside of the for-loop otherwise you keep overwriting the whole file
    blob.upload_from_string(data=json.dumps(data), content_type='application/json')

    return True


@stub.function(
    image=image,
    timeout=86400,
    volumes={"/dna_files": volume}
)
def parse():
    print("Starting Parser...")
    parse_data()


@stub.function(
    image=image,
    timeout=86400,
    volumes={"/dna_files": volume}
)
def get_data():
    print(f"Collecting data from {BASE_URL}...")
    get_ids_list("nucleotide", "txid9606%5bOrganism:noexp%5d")
