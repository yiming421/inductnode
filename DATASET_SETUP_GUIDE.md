# InductNode Dataset Upload and Download Guide

This guide helps you upload your InductNode dataset to Google Drive and enable automatic downloading for others.

## 📁 File Structure

The solution includes these key files:
- `scripts/prepare_dataset_for_upload.py` - Compress dataset for upload
- `scripts/download_dataset_from_gdrive.py` - Manual download script
- `src/dataset_config.py` - Configuration for file IDs
- `src/dataset_downloader.py` - Unified download utilities
- `src/data.py` & `src/data_link.py` - Auto-download integration

## Step 1: Prepare Dataset for Upload

Run the preparation script to compress your dataset:

```bash
cd /home/maweishuo/inductnode
python scripts/prepare_dataset_for_upload.py
```

This will:
- Compress your `dataset/` directory into a `.tar.gz` file (~6.4GB → ~2-3GB compressed)
- Create upload instructions in `UPLOAD_INSTRUCTIONS.md`

## Step 2: Upload to Google Drive

### Manual Upload (Recommended)
1. Go to [Google Drive](https://drive.google.com)
2. Click "New" → "File upload"
3. Select the compressed file (e.g., `inductnode_dataset_20250716_123456.tar.gz`)
4. Wait for upload to complete (may take 10-30 minutes depending on connection)

### Get Shareable Link
1. Right-click on the uploaded file
2. Select "Get link"
3. Set permissions to "Anyone with the link can view"
4. Copy the sharing link

### Extract File ID
From a sharing link like:
```
https://drive.google.com/file/d/1ABC123xyz789DEF/view?usp=sharing
```
Extract the file ID: `1ABC123xyz789DEF`

## Step 3: Configure Auto-Download

Edit `src/dataset_config.py` and update the file ID:

```python
GDRIVE_DATASET_FILE_ID = "1ABC123xyz789DEF"  # Your actual file ID
```

## Step 4: Test Download

Others can now download your dataset automatically by:

```bash
# Clone your repository
git clone https://github.com/yiming421/inductnode.git
cd inductnode

# Run any script that needs datasets - it will auto-download
python scripts/pfn.py --datasets Cora
```

Or manually download with:

```bash
python scripts/download_dataset_from_gdrive.py
```

## For End Users

### Automatic Download
When users run scripts that need datasets, they will be automatically downloaded:
- First time running any dataset-dependent script
- If dataset directory is missing or appears incomplete
- Completely transparent to the user

### Manual Download
If auto-download fails, users can:

1. **Download using the script:**
   ```bash
   python scripts/download_dataset_from_gdrive.py
   ```

2. **Download directly with gdown:**
   ```bash
   pip install gdown
   gdown https://drive.google.com/uc?id=YOUR_FILE_ID
   tar -xzf inductnode_dataset_*.tar.gz
   ```

## File Structure After Download

After download and extraction, you should have:

```
inductnode/
├── dataset/
│   ├── Cora/
│   ├── Flickr/
│   ├── Amazon/
│   ├── Coauthor/
│   ├── Reddit/
│   ├── CitationFull/
│   ├── Airports/
│   ├── AttributedGraph/
│   ├── ogbn_arxiv/
│   └── ...
├── src/
├── scripts/
└── ...
```

## Code Integration

The auto-download is integrated into:
- `src/data.py` - Node classification datasets
- `src/data_link.py` - Link prediction datasets
- All PyTorch Geometric dataset loaders will trigger auto-download if needed

## Troubleshooting

### Upload Issues
- **File too large**: Google Drive free accounts have 15GB limit
- **Slow upload**: Try uploading during off-peak hours
- **Upload failed**: Check internet connection, try again

### Download Issues
- **File ID not set**: Update `GDRIVE_DATASET_FILE_ID` in `src/dataset_config.py`
- **Permission denied**: Ensure Google Drive file is set to "Anyone with the link can view"
- **gdown failed**: Try `pip install --upgrade gdown`
- **Extraction failed**: Check available disk space (need ~7GB free)

### Network Issues
- **Slow download**: Google Drive may throttle large downloads
- **Connection timeout**: Script will retry automatically
- **Partial download**: Delete partial files and retry

## Benefits of This Approach

✅ **Unified download logic** - No code duplication
✅ **Automatic installation** - gdown installs automatically if needed  
✅ **Error handling** - Robust error handling and recovery
✅ **File validation** - Checks file sizes and structure
✅ **Zero configuration** - Works out of the box once file ID is set
✅ **Backwards compatible** - Existing code continues to work

## Current Status

- ✅ Dataset compression script ready
- ✅ Unified download utilities implemented
- ✅ Auto-download integrated in data loaders
- ✅ Error handling and validation
- ⏳ **Need to upload dataset and update file ID**
- ⏳ **Need to test complete workflow**

## Next Steps

1. **Run the preparation script:**
   ```bash
   python scripts/prepare_dataset_for_upload.py
   ```

2. **Upload to Google Drive** (follow instructions above)

3. **Update configuration:**
   ```bash
   # Edit src/dataset_config.py
   GDRIVE_DATASET_FILE_ID = "your_file_id_here"
   ```

4. **Test the download:**
   ```bash
   # Remove existing dataset to test
   rm -rf dataset/
   # Run any script - should auto-download
   python scripts/pfn.py --datasets Cora
   ```

5. **Share your repository** - datasets will auto-download for all users!
