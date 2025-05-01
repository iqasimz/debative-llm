# scripts/upload_hf_model.py

from huggingface_hub import Repository
import os

# 1. Point to your real local model folder
local_folder = os.path.expanduser(
    "/Users/qz/projects/debative-llm/models/role_student_v2_highconf"
)

# 2. Your Hugging Face repo
repo_id = "iqasimz/role_student_v2_highconf"

# 3. Clone the remote repo locally (or reuse if exists)
repo = Repository(
    local_dir="/tmp/role_student_v2_hf",
    clone_from=f"https://huggingface.co/{repo_id}",
)

# 4. Copy all files from your local folder into that repo working dir
for fname in os.listdir(local_folder):
    src = os.path.join(local_folder, fname)
    dst = os.path.join(repo.local_dir, fname)
    if os.path.isfile(src):
        # overwrite the file in the HF repo clone
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

# 5. Commit & push
repo.push_to_hub(commit_message="Upload full model artifacts")
print("✅ Model files pushed to Hugging Face Hub")