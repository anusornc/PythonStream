from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')
