from huggingface_hub import snapshot_download

# Downloads everything into a local folder
path = snapshot_download(repo_id="Sashank-810/crisisnet-dataset", repo_type="dataset")
print("Downloaded to:", path)