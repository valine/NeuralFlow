# Neural Flow

This is a python script to plot the intermediate layer output of Mistral 7B. When you run the script it will produce a 512x256 image representing the output at every layer of the model. 

![Alt text](https://private-user-images.githubusercontent.com/14074844/304293703-ab939cc2-a5fa-4a1a-8e45-bc2b5741f0e1.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc3OTg2MDUsIm5iZiI6MTcwNzc5ODMwNSwicGF0aCI6Ii8xNDA3NDg0NC8zMDQyOTM3MDMtYWI5MzljYzItYTVmYS00YTFhLThlNDUtYmMyYjU3NDFmMGUxLmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAyMTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMjEzVDA0MjUwNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJjN2MxZTNmNGM3OTFiY2FhMmZlZTlhNWM3YmY1ZjUyNTVjYmI0ZmZkY2FiZWY1Y2I3YjJhZDU5OWEyNzRhYzkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.3ob2VPnlhDhl-JO0Fw8sWScr7VEmg1ZmojFLgF9wKz8)

# Constants
There are two file paths you will want to change before running the script:

```
model_folder = "/models/OpenHermes-2.5-Mistral-7B"
image_output_folder = "/home/username/Desktop/"
```

This is self explanitory, but set the model folder to the location of your Mistral 7B, and the image output folder to the path you'd like to save your image.
