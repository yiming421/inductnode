# Dataset Upload Instructions

## File Information
- **File**: inductnode_dataset_20250723_122123.tar.gz
- **Size**: 7661.1 MB
- **Created**: 2025-07-23 13:14:35

## Upload to Google Drive

### Step 1: Upload to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Click "New" → "File upload"
3. Select the file: `inductnode_dataset_20250723_122123.tar.gz`
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
tar -xzf inductnode_dataset_20250723_122123.tar.gz
```
