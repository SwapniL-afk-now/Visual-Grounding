import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage
import os
import sys

def display_diagnostic(image_path="grounding_diagnostic.png"):
    """
    Standalone utility to force display of the diagnostic plot in interactive environments.
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Run the diagnostic script first.")
        return

    print(f"Loading diagnostic results from {image_path}...")
    
    # Method 1: IPython Display (Most reliable in Kaggle/Jupyter)
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            display(IPImage(filename=image_path))
            print("Displayed via IPython.display")
            return
    except Exception as e:
        print(f"IPython display failed: {e}")

    # Method 2: Matplotlib Fallback
    try:
        img = plt.imread(image_path)
        plt.figure(figsize=(20, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        print("Displayed via Matplotlib")
    except Exception as e:
        print(f"Matplotlib display failed: {e}")
        print(f"Please manually open the file: {os.path.abspath(image_path)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "grounding_diagnostic.png"
    display_diagnostic(path)
