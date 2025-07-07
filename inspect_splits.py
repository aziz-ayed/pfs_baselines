# inspect_split.py
import torch
import sys
import pathlib # We need this for the unpickler to recognize Path objects

def main():
    # Get the file path from the command line
    if len(sys.argv) != 2:
        print("Usage: python inspect_split.py <path_to_splits.pt>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"--- Inspecting file: {file_path} ---")
    
    try:
        # Load the object from the file, allowing it to unpickle Path objects
        loaded_object = torch.load(file_path, weights_only=False)
        
        # Print the type of the top-level object
        print(f"\nType of loaded object: {type(loaded_object)}")
        
        # Print more details based on the type
        if isinstance(loaded_object, dict):
            print("\nObject is a DICTIONARY. Keys found:")
            print(list(loaded_object.keys()))
        elif isinstance(loaded_object, tuple):
            print(f"\nObject is a TUPLE with {len(loaded_object)} elements.")
        else:
            print("\nObject is of an unexpected type.")

    except Exception as e:
        print(f"\nAn error occurred while loading the file: {e}")
        
    print("\n--- Inspection complete ---")

if __name__ == "__main__":
    main()
