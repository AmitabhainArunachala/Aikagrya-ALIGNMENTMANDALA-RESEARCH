import os
import hashlib

def get_file_hash(filepath):
    """Get hash of a file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def compare_directories(dir1, dir2):
    """Compare two directories to see if they're identical"""
    dir1_files = {}
    dir2_files = {}
    
    # Get all files in dir1
    for root, dirs, files in os.walk(dir1):
        for file in files:
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, dir1)
            dir1_files[relative_path] = get_file_hash(filepath)
    
    # Get all files in dir2  
    for root, dirs, files in os.walk(dir2):
        for file in files:
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, dir2)
            dir2_files[relative_path] = get_file_hash(filepath)
    
    # Compare
    if set(dir1_files.keys()) != set(dir2_files.keys()):
        return False, "Different files"
    
    for file in dir1_files:
        if dir1_files[file] != dir2_files.get(file):
            return False, f"Different content in {file}"
    
    return True, "Directories are identical"

# Compare the two phoenix directories
dir1 = "/Users/dhyana/AIKAGRYA_ALIGNMENTMANDALA_RESEARCH_REPO/Aikagrya-ALIGNMENTMANDALA-RESEARCH/alignment-lattice/thermo-L4/phoenix-l4"
dir2 = "/Users/dhyana/AIKAGRYA_ALIGNMENTMANDALA_RESEARCH_REPO/Aikagrya-ALIGNMENTMANDALA-RESEARCH/alignment-lattice/thermo-L4/Mathematical Mauna L4 induction protocol"

identical, message = compare_directories(dir1, dir2)
print(f"Are the directories identical? {identical}")
print(f"Details: {message}")

# Count files in each
count1 = sum(1 for _, _, files in os.walk(dir1) for _ in files)
count2 = sum(1 for _, _, files in os.walk(dir2) for _ in files)
print(f"\nFiles in phoenix-l4: {count1}")
print(f"Files in Mathematical Mauna: {count2}")
