"""
Unified dataset download utilities for InductNode
This module provides common functionality for downloading datasets from Google Drive
"""
import os
import sys
import subprocess
import tarfile
from typing import Optional

def install_gdown() -> bool:
    """
    Install gdown if not available
    
    Returns:
        True if gdown is available or successfully installed, False otherwise
    """
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown for dataset download...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
            import gdown
            print("gdown installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install gdown: {e}")
            return False

def download_from_gdrive(file_id: str, output_path: str, min_size_mb: float = 10) -> bool:
    """
    Download a file from Google Drive using gdown
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path where the file should be saved
        min_size_mb: Minimum expected file size in MB for validation
    
    Returns:
        True if download successful, False otherwise
    """
    if not install_gdown():
        return False
    
    try:
        import gdown
        
        download_url = f'https://drive.google.com/uc?id={file_id}'
        print(f"Downloading from Google Drive...")
        print(f"File ID: {file_id}")
        print(f"Output: {output_path}")
        
        gdown.download(download_url, output_path, quiet=False)
        
        # Verify download
        if not os.path.exists(output_path):
            print("Download failed - file not found")
            return False
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if file_size_mb < min_size_mb:
            print(f"Downloaded file appears too small ({file_size_mb:.1f} MB). Download may have failed.")
            return False
        
        print(f"Download complete! File size: {file_size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def extract_tarfile(compressed_file: str, extract_to: str) -> bool:
    """
    Extract a tar.gz file
    
    Args:
        compressed_file: Path to the compressed file
        extract_to: Directory to extract to
    
    Returns:
        True if extraction successful, False otherwise
    """
    print(f"Extracting {compressed_file} to {extract_to}...")
    
    try:
        with tarfile.open(compressed_file, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print("Extraction complete!")
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def verify_dataset_structure(dataset_path: str, expected_dirs: list = None) -> bool:
    """
    Verify that the dataset has the expected structure
    
    Args:
        dataset_path: Path to the dataset directory
        expected_dirs: List of expected subdirectories (optional)
    
    Returns:
        True if dataset structure is valid, False otherwise
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found: {dataset_path}")
        return False
    
    if expected_dirs is None:
        expected_dirs = ['Cora', 'Flickr', 'Amazon', 'Coauthor', 'Reddit']
    
    found_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            found_dirs.append(item)
    
    print(f"Found dataset directories: {', '.join(found_dirs)}")
    
    # Check if we have at least some expected directories
    common_dirs = set(expected_dirs) & set(found_dirs)
    if len(common_dirs) > 0:
        print(f"Dataset verification passed! Found {len(common_dirs)} expected directories.")
        return True
    else:
        print("Warning: No expected dataset directories found")
        return False

def download_and_extract_dataset(file_id: str, project_root: str, cleanup: bool = True) -> bool:
    """
    Download and extract the complete dataset from Google Drive
    
    Args:
        file_id: Google Drive file ID
        project_root: Project root directory
        cleanup: Whether to remove the compressed file after extraction
    
    Returns:
        True if successful, False otherwise
    """
    if not file_id:
        print("File ID not provided")
        return False
    
    compressed_file = os.path.join(project_root, 'inductnode_dataset.tar.gz')
    
    # Download
    if not download_from_gdrive(file_id, compressed_file, min_size_mb=10):
        return False
    
    # Extract
    if not extract_tarfile(compressed_file, project_root):
        return False
    
    # Verify
    dataset_path = os.path.join(project_root, 'dataset')
    if not verify_dataset_structure(dataset_path):
        print("Warning: Dataset verification failed")
    
    # Cleanup
    if cleanup:
        try:
            os.remove(compressed_file)
            print("Cleaned up compressed file")
        except Exception as e:
            print(f"Could not remove compressed file: {e}")
    
    return True
