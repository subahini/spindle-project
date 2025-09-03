

"""
This file Downloads a spindle detection JSON label file from lakeFS  and save it localy  """
import boto3
import yaml
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def download_label_file():
    config = load_config()
    lakefs = config["lakefs"]
    label_cfg = config["labels"]
   # connecting to  lakefs 
    s3 = boto3.client("s3", endpoint_url=lakefs["endpoint"])

    repo = lakefs["repo"]
    key = label_cfg["key"]
    local_path = label_cfg["local"]

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
     # Downloading
    print(f" Downloading label JSON from lakeFS: s3://{repo}/{key}")
    response = s3.get_object(Bucket=repo, Key=key)
    # saving
    with open(local_path, "wb") as f:
        f.write(response["Body"].read())
    
    print(f"Saved to: {local_path}")

if __name__ == "__main__":
    download_label_file()
