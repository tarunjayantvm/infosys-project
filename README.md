# AI Vision Extract

A professional Flask web application for AI-powered object detection and masking.

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Add sample images:
   - Place your sample original image in `static/images/sample-original.jpg`
   - Place your sample masked image in `static/images/sample-masked.jpg`

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Features

- Beautiful, professional UI with gradient design
- Sample image showcase (original and masked)
- File upload functionality with drag-and-drop support
- Responsive design for all screen sizes
- Ready for AI masking integration

## Project Structure

```
project/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   ├── js/
│   │   └── script.js     # Frontend logic
│   ├── images/           # Sample images
│   └── uploads/          # User uploaded images
```
