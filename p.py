import easyocr, shutil, os, pathlib
dest = pathlib.Path('models/easyocr')
dest.mkdir(parents=True, exist_ok=True)   # parents=True creates 'models' if missing
reader = easyocr.Reader(['en'])
shutil.move(os.path.expanduser('~/.EasyOCR'), dest)
print('Weights saved to ./models/easyocr')
