import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor

def create_github_friendly_notebook(input_path, output_path):
    # Read the notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Clear outputs
    cop = ClearOutputPreprocessor()
    nb, _ = cop.preprocess(nb, {})
    
    # Remove all widget metadata
    for cell in nb.cells:
        if hasattr(cell, 'metadata'):
            if 'widgets' in cell.metadata:
                del cell.metadata['widgets']
            # Also remove other problematic metadata
            if 'execution' in cell.metadata:
                del cell.metadata['execution']
    
    # Clean notebook metadata
    if 'widgets' in nb.metadata:
        del nb.metadata['widgets']
    
    # Save the cleaned notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Create a GitHub-friendly version
create_github_friendly_notebook(
    'crop_desease_detection.ipynb', 
    'crop_desease_detection_github.ipynb'
)