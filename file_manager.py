#!/usr/bin/env python3
"""
File Manager Utility
Handles file existence checks and prevents duplicate file creation
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations with existence checks and duplication prevention"""
    
    def __init__(self, base_directory: str = "."):
        """
        Initialize file manager
        
        Args:
            base_directory: Base directory for file operations
        """
        self.base_directory = base_directory
        self.file_registry = {}  # Track created files
        self.load_file_registry()
    
    def load_file_registry(self):
        """Load existing file registry"""
        registry_file = os.path.join(self.base_directory, ".file_registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    self.file_registry = json.load(f)
                logger.info(f"Loaded file registry with {len(self.file_registry)} entries")
            except Exception as e:
                logger.error(f"Error loading file registry: {e}")
                self.file_registry = {}
    
    def save_file_registry(self):
        """Save file registry to disk"""
        registry_file = os.path.join(self.base_directory, ".file_registry.json")
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.file_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving file registry: {e}")
    
    def get_file_hash(self, filepath: str) -> Optional[str]:
        """Calculate MD5 hash of file content"""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash for {filepath}: {e}")
            return None
    
    def file_exists_and_matches(self, filepath: str, content_hash: str = None) -> bool:
        """
        Check if file exists and optionally matches content hash
        
        Args:
            filepath: Path to check
            content_hash: Optional content hash to verify
            
        Returns:
            True if file exists and matches (if hash provided)
        """
        if not os.path.exists(filepath):
            return False
        
        if content_hash:
            file_hash = self.get_file_hash(filepath)
            return file_hash == content_hash
        
        return True
    
    def safe_create_directory(self, directory: str) -> bool:
        """
        Safely create directory if it doesn't exist
        
        Args:
            directory: Directory path to create
            
        Returns:
            True if directory exists or was created successfully
        """
        if os.path.exists(directory):
            logger.info(f"Directory already exists: {directory}")
            return True
        
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    def safe_save_json(self, data: Dict, filepath: str, 
                      check_existing: bool = True, 
                      content_hash: str = None) -> bool:
        """
        Safely save JSON data to file, checking for existing content
        
        Args:
            data: Data to save
            filepath: Target file path
            check_existing: Whether to check for existing file
            content_hash: Optional content hash for verification
            
        Returns:
            True if file was saved or already exists with matching content
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not self.safe_create_directory(directory):
            return False
        
        # Check if file already exists
        if check_existing and self.file_exists_and_matches(filepath, content_hash):
            logger.info(f"File already exists with matching content: {filepath}")
            return True
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Calculate and store hash
            file_hash = self.get_file_hash(filepath)
            if file_hash:
                self.file_registry[filepath] = {
                    'hash': file_hash,
                    'created': datetime.now().isoformat(),
                    'size': os.path.getsize(filepath)
                }
                self.save_file_registry()
            
            logger.info(f"Saved JSON file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {e}")
            return False
    
    def safe_save_csv(self, data: Any, filepath: str, 
                     check_existing: bool = True) -> bool:
        """
        Safely save CSV data to file
        
        Args:
            data: Data to save (DataFrame or similar)
            filepath: Target file path
            check_existing: Whether to check for existing file
            
        Returns:
            True if file was saved or already exists
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not self.safe_create_directory(directory):
            return False
        
        # Check if file already exists
        if check_existing and os.path.exists(filepath):
            logger.info(f"CSV file already exists: {filepath}")
            return True
        
        try:
            if hasattr(data, 'to_csv'):
                data.to_csv(filepath, index=True)
            else:
                # Handle other data types
                import pandas as pd
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
                df.to_csv(filepath, index=True)
            
            # Store file info
            self.file_registry[filepath] = {
                'hash': self.get_file_hash(filepath),
                'created': datetime.now().isoformat(),
                'size': os.path.getsize(filepath)
            }
            self.save_file_registry()
            
            logger.info(f"Saved CSV file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving CSV file {filepath}: {e}")
            return False
    
    def safe_save_text(self, content: str, filepath: str, 
                      check_existing: bool = True) -> bool:
        """
        Safely save text content to file
        
        Args:
            content: Text content to save
            filepath: Target file path
            check_existing: Whether to check for existing file
            
        Returns:
            True if file was saved or already exists
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not self.safe_create_directory(directory):
            return False
        
        # Check if file already exists
        if check_existing and os.path.exists(filepath):
            logger.info(f"Text file already exists: {filepath}")
            return True
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Store file info
            self.file_registry[filepath] = {
                'hash': self.get_file_hash(filepath),
                'created': datetime.now().isoformat(),
                'size': os.path.getsize(filepath)
            }
            self.save_file_registry()
            
            logger.info(f"Saved text file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving text file {filepath}: {e}")
            return False
    
    def get_file_info(self, filepath: str) -> Optional[Dict]:
        """Get information about a file"""
        if not os.path.exists(filepath):
            return None
        
        try:
            stat = os.stat(filepath)
            return {
                'path': filepath,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'hash': self.get_file_hash(filepath)
            }
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return None
    
    def list_created_files(self, pattern: str = None) -> List[str]:
        """
        List files created by this manager
        
        Args:
            pattern: Optional pattern to filter files
            
        Returns:
            List of file paths
        """
        files = list(self.file_registry.keys())
        
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        
        return files
    
    def cleanup_old_files(self, max_age_hours: int = 24, 
                         file_pattern: str = None) -> int:
        """
        Clean up old files
        
        Args:
            max_age_hours: Maximum age in hours
            file_pattern: Optional pattern to match files
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        files_to_check = self.list_created_files(file_pattern)
        
        for filepath in files_to_check:
            if not os.path.exists(filepath):
                # Remove from registry if file doesn't exist
                self.file_registry.pop(filepath, None)
                continue
            
            try:
                file_time = os.path.getmtime(filepath)
                if file_time < cutoff_time:
                    os.remove(filepath)
                    self.file_registry.pop(filepath, None)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {e}")
        
        if cleaned_count > 0:
            self.save_file_registry()
        
        return cleaned_count
    
    def get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory}: {e}")
        
        return total_size
    
    def print_summary(self):
        """Print file manager summary"""
        print("📁 FILE MANAGER SUMMARY")
        print("=" * 50)
        print(f"📂 Base Directory: {self.base_directory}")
        print(f"📊 Total Files Tracked: {len(self.file_registry)}")
        
        if self.file_registry:
            total_size = sum(info.get('size', 0) for info in self.file_registry.values())
            print(f"💾 Total Size: {total_size / (1024*1024):.2f} MB")
            
            # Group by directory
            directories = {}
            for filepath in self.file_registry.keys():
                directory = os.path.dirname(filepath) or "."
                if directory not in directories:
                    directories[directory] = 0
                directories[directory] += 1
            
            print(f"📁 Directories: {len(directories)}")
            for directory, count in sorted(directories.items()):
                print(f"   {directory}: {count} files")
        
        print("=" * 50)

def main():
    """Example usage of file manager"""
    print("🚀 FILE MANAGER UTILITY")
    print("=" * 50)
    
    # Create file manager
    fm = FileManager()
    
    # Example operations
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'message': 'Test data',
        'values': [1, 2, 3, 4, 5]
    }
    
    # Safe save operations
    fm.safe_save_json(test_data, 'test_output/test_data.json')
    fm.safe_save_json(test_data, 'test_output/test_data.json')  # Should skip
    
    # Print summary
    fm.print_summary()
    
    # Cleanup example
    cleaned = fm.cleanup_old_files(max_age_hours=1)
    if cleaned > 0:
        print(f"Cleaned up {cleaned} old files")

if __name__ == "__main__":
    main() 