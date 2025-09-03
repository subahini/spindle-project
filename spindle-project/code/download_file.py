
"""
This file Downloads a raw EEG .edf file from lakeFS.
Uses boto3 with endpoint and path defined in config/pipeline.yaml. 
they save the file to local file 
"""



import boto3
import yaml
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def download_from_lakefs():
    cfg = load_config()
    lakefs = cfg["lakefs"]
    paths = cfg["paths"]

    # Connect to lakeFS using boto3
    s3 = boto3.client("s3", endpoint_url=lakefs["endpoint"])

    # Download the EEG file
    print(f"Downloading from lakeFS: s3://{lakefs['repo']}/{lakefs['key']}")
    response = s3.get_object(Bucket=lakefs["repo"], Key=lakefs["key"])

    # Create local folder and save file
    raw_path = Path(paths["raw_local"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    #   saving to local file 
    with open(raw_path, "wb") as f:
        f.write(response["Body"].read())

    print(f" File saved to: {raw_path}")

if __name__ == "__main__":
    download_from_lakefs()
