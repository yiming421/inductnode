#!/usr/bin/env python3
"""
Simplified script to prepare dataset for Google Drive upload using gdown
"""
import os
import sys
import tarfile
from datetime import datetime

def get_project_root():
    """Get the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compress_dataset():
    """Compress the dataset directory"""
    project_root = get_project_root()
    dataset_dir = os.path.join(project_root, 'dataset')
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    # Create compressed filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compressed_file = os.path.join(project_root, f'inductnode_dataset_{timestamp}.tar.gz')
    
    print(f"Compressing dataset to {compressed_file}...")
    print("This may take a few minutes...")
    
    try:
        with tarfile.open(compressed_file, 'w:gz') as tar:
            # Add with progress indication
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_root)
                    tar.add(file_path, arcname=arcname)
                    print(f"Added: {arcname}", end='\r')
        
        print("\nCompression complete!")
        
        # Check compressed file size
        size_mb = os.path.getsize(compressed_file) / (1024 * 1024)
        print(f"Compressed file: {compressed_file}")
        print(f"File size: {size_mb:.1f} MB")
        
        return compressed_file
    except Exception as e:
        print(f"Compression failed: {e}")
        return None

def create_upload_instructions(compressed_file):
    """Create instructions for manual upload"""
    project_root = get_project_root()
    instructions_file = os.path.join(project_root, 'UPLOAD_INSTRUCTIONS.md')
    
    file_size_mb = os.path.getsize(compressed_file) / (1024 * 1024)
    
    content = f"""# Dataset Upload Instructions

## File Information
- **File**: {os.path.basename(compressed_file)}
- **Size**: {file_size_mb:.1f} MB
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Upload to Google Drive

### Step 1: Upload to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Click "New" → "File upload"
3. Select the file: `{os.path.basename(compressed_file)}`
4. Wait for upload to complete

### Step 2: Get Shareable Link
1. Right-click on the uploaded file
2. Select "Get link"
3. Set permissions to "Anyone with the link can view"
4. Copy the sharing link

### Step 3: Extract File ID
From a link like: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
Extract the FILE_ID part.

### Step 4: Update Code
Add the FILE_ID to the dataset download configuration in your code.

## Alternative: Use gdown for Upload
```bash
# Install gdown
pip install gdown[requests]

# Upload (if you have gdown upload capabilities)
# Note: gdown primarily supports download, not upload
```

## For Users to Download
Once uploaded, users can download with:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID
tar -xzf {os.path.basename(compressed_file)}
```
"""
    
    with open(instructions_file, 'w') as f:
        f.write(content)
    
    print(f"Upload instructions saved to: {instructions_file}")

def main():
    print("=== InductNode Dataset Preparation for Google Drive ===")
    
    # Compress dataset
    compressed_file = compress_dataset()
    if not compressed_file:
        return 1
    
    # Create upload instructions
    create_upload_instructions(compressed_file)
    
    print("\n=== Preparation Complete! ===")
    print(f"Compressed file ready: {compressed_file}")
    print("Please follow the instructions in UPLOAD_INSTRUCTIONS.md to upload to Google Drive.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
