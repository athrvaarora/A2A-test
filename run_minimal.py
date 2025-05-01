import subprocess
import sys
import os

def main():
    print("A2A Demo Minimal Runner")
    print("=======================")
    print("\nThis script provides a minimal way to explore the A2A Demo without installing all dependencies.")
    
    # Check the directory structure
    if not os.path.exists("a2a_demo"):
        print("Error: Cannot find a2a_demo directory. Make sure you're running this from the right location.")
        return
        
    # Show what we can do
    print("\nAvailable options:")
    print("1. Examine project structure")
    print("2. Show README")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        print("\nProject structure:")
        for root, dirs, files in os.walk("a2a_demo", topdown=True):
            level = root.replace("a2a_demo", "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = " " * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
    elif choice == "2":
        try:
            with open("a2a_demo/README.md", "r") as f:
                print("\nREADME contents:")
                print(f.read())
        except:
            print("Could not open README.md")
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 