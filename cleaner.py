import json
import nbformat

def clean_notebook(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Remove widget metadata from each cell
    for cell in nb.cells:
        if 'metadata' in cell and 'widgets' in cell.metadata:
            # Either remove widgets entirely
            del cell.metadata['widgets']
            # Or add state if you want to keep widgets
            # cell.metadata['widgets']['state'] = {}
    
    # Also check notebook-level metadata
    if 'metadata' in nb and 'widgets' in nb.metadata:
        # Either remove widgets entirely
        del nb.metadata['widgets']
        # Or add state if you want to keep widgets
        # nb.metadata['widgets']['state'] = {}
    
    # Save the cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Cleaned notebook saved to {notebook_path}")

# Clean your notebook
clean_notebook('crop_desease_detection.ipynb')