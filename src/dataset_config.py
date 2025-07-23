"""
Configuration file for dataset download URLs and file IDs
Update these values after uploading your dataset to Google Drive
"""

# Google Drive Configuration
GDRIVE_DATASET_FILE_ID = None  # Update this with your Google Drive file ID after upload

# Alternative download URLs (if you use other platforms)
ALTERNATIVE_DOWNLOAD_URLS = {
    # Example: "dataset_name": "download_url"
    # "flickr": "https://your-alternative-url.com/flickr.tar.gz"
}

# Dataset verification - expected directories in the dataset folder
EXPECTED_DATASET_DIRS = [
    'Cora', 'Flickr', 'Amazon', 'Coauthor', 'Reddit', 
    'CitationFull', 'Airports', 'AttributedGraph', 
    'WikiCS', 'FacebookPagePage', 'ogbn_arxiv'
]

# Minimum number of directories required to consider dataset complete
MIN_DIRS_FOR_COMPLETE = 5

def get_gdrive_download_url():
    """Get the Google Drive download URL"""
    if GDRIVE_DATASET_FILE_ID:
        return f"https://drive.google.com/uc?id={GDRIVE_DATASET_FILE_ID}"
    return None

def is_dataset_complete(dataset_root):
    """Check if the dataset appears to be complete"""
    import os
    
    if not os.path.exists(dataset_root):
        return False
    
    existing_dirs = [d for d in os.listdir(dataset_root) 
                    if os.path.isdir(os.path.join(dataset_root, d))]
    
    # Check if we have at least the minimum required directories
    common_dirs = set(EXPECTED_DATASET_DIRS) & set(existing_dirs)
    return len(common_dirs) >= MIN_DIRS_FOR_COMPLETE
