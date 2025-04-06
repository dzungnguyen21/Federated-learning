import os
import sys
import shutil

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def cleanup_data_files():
    """Remove large dataset files that shouldn't be pushed to GitHub"""
    
    # Define paths to check and clean
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dirs = [
        os.path.join(root_dir, 'data'),
        os.path.join(root_dir, 'Data', 'cifar-10-batches-py'),
        os.path.join(root_dir, 'Data', 'MNIST')
    ]
    
    # Files to specifically remove
    specific_files = [
        os.path.join(root_dir, 'Data', 'cifar-10-python.tar.gz')
    ]
    
    # Remove directories
    for directory in data_dirs:
        if os.path.exists(directory):
            print(f"Removing directory: {directory}")
            shutil.rmtree(directory)
    
    # Remove specific files
    for file_path in specific_files:
        if os.path.exists(file_path):
            print(f"Removing file: {file_path}")
            os.remove(file_path)
    
    print("\nCleanup completed successfully!")
    print("Your repository is now ready to push to GitHub without large dataset files.")
    print("\nNote: Next time you run the code, datasets will be automatically downloaded again.")

if __name__ == "__main__":
    # Ask for confirmation
    print("This script will remove large dataset files from your repository to prepare for GitHub push.")
    choice = input("Do you want to continue? (y/n): ")
    
    if choice.lower() == 'y':
        cleanup_data_files()
    else:
        print("Cleanup canceled.")