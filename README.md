# HOW TO RUN
Follow these steps to launch the application:

1. Open a Terminal:
- On Windows: Press Win + R, type cmd, and press Enter.
- On macOS/Linux: Open your terminal application.


2. Navigate to the Project Directory:
```bash
cd /path/to/EmongWeb
```
Replace /path/to/EmongWeb with the actual path to the project folder.

3. Install Virtual Environment (if you installed before, skip this step):
```bash
pip install virtualenv
```

4. Create Virtual Environment:
```bash
python -m venv venv
```

5. Activate the Virtual Environment:
- On Mac:
```bash
source venv/bin/activate
```
- On Windows:
```bash
venv\Scripts\activate
```

6. Install Dependencies (if not done):
```bash
pip install -r requirements.txt
```

7. Start the Flask Server:
```bash
python app.py
```

The server will run on localhost in debug mode.
Look for logs confirming the YOLO model loading and server startup.
