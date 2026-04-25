import os
import sys
import time

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Please install faster-whisper first: pip install faster-whisper")
    sys.exit(1)

# Use a mirror if you frequently have connection issues with Hugging Face
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# The default model used in the app
model_id = "Systran/faster-whisper-base"

print(f"Downloading model {model_id} from Hugging Face Hub (using hf-mirror)...")
print("This might take a few minutes depending on your internet connection.")

max_retries = 5
for attempt in range(max_retries):
    try:
        path = snapshot_download(
            repo_id=model_id,
            resume_download=True,  # Automatically resumes partial downloads
            local_files_only=False
        )
        print("\n✅ Download completed successfully!")
        print(f"Model saved at: {path}")
        break
    except Exception as e:
        print(f"\n❌ Error downloading model (Attempt {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Please try running this script again to resume the download.")
            sys.exit(1)
