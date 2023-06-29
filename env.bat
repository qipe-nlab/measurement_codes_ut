py -m venv .venv
cd .venv/Scripts
call "activate.bat"
py -m pip install --upgrade pip
cd ../..
pip install -r requirements.txt