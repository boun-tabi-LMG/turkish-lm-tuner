## Library and documentation instructions
Follow these steps to build library and documentation

### Install required packages
```bash
pip install setuptools wheel build
```
### Build the library:
```
bash
cd ~/turkish-lm-tuner
python -m build
```
This command will compile the project into a distributable format.

### Install Documentation Requirements
First, ensure that the documentation requirements specified in env.yml are installed.

### Build and Serve Documentation
Use mkdocs to build and serve the documentation:
```
bash
mkdocs serve
```
After executing this command, you can view the documentation by visiting http://127.0.0.1:8000/boun-tabi-lmt/turkish-lm-tuner/ in your web browser.


