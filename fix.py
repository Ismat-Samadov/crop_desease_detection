import json
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def fix_notebook_widgets(input_path, output_path=None):
    """
    Fix notebook widget metadata issues
    """
    if output_path is None:
        output_path = input_path
    
    # Read the notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Fix cell-level widget metadata
    for cell in nb.cells:
        if hasattr(cell, 'metadata') and 'widgets' in cell.metadata:
            if isinstance(cell.metadata['widgets'], dict):
                if 'state' not in cell.metadata['widgets']:
                    cell.metadata['widgets']['state'] = {}
            else:
                # If widgets is not a dict, remove it
                del cell.metadata['widgets']
    
    # Fix notebook-level widget metadata
    if 'widgets' in nb.metadata:
        if isinstance(nb.metadata['widgets'], dict):
            if 'state' not in nb.metadata['widgets']:
                nb.metadata['widgets']['state'] = {}
        else:
            # If widgets is not a dict, remove it
            del nb.metadata['widgets']
    
    # Remove any empty widget metadata
    for cell in nb.cells:
        if hasattr(cell, 'metadata') and 'widgets' in cell.metadata:
            if not cell.metadata['widgets'] or (
                len(cell.metadata['widgets']) == 1 and 
                'state' in cell.metadata['widgets'] and 
                not cell.metadata['widgets']['state']
            ):
                del cell.metadata['widgets']
    
    # Write the fixed notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Fixed notebook saved to {output_path}")

# Fix your notebook
fix_notebook_widgets('crop_desease_detection.ipynb')