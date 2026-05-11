#!/bin/bash
cd "/Users/leo_httl/Downloads/DL/Final Project"
rm -rf dl4ai-env
python3 -m venv dl4ai-env
source dl4ai-env/bin/activate
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install jupyter notebook ipykernel
pip install ta fastapi uvicorn streamlit
python -m ipykernel install --user --name=dl4ai-env --display-name="DL4AI"
echo ""
echo "=============================="
echo "Setup complete! Now run:"
echo "  cd /Users/leo_httl/Downloads/DL/Final\ Project"
echo "  source dl4ai-env/bin/activate"
echo "  jupyter notebook"
echo "=============================="
