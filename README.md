# Neural Flow

This is a python script to plot the intermediate layer output of Mistral 7B. When you run the script it will produce a 512x256 image representing the output at every layer of the model. 

The idea is pretty simple: collect the output tensors at every layer, normalize between zero and one, and plot those values as a heat map. The resulting image has a surprising amount of structure, and I have found it enormously helpful when fine-tuning models as a way to visually inspect the output.
This is the visualization from mistral 7B out of the box with no fine tuning.

![probe_results_layers_20240212_221145](https://github.com/valine/NeuralFlow/assets/14074844/ab939cc2-a5fa-4a1a-8e45-bc2b5741f0e1)


# Constants
There are two file paths you will want to change before running the script:

```
model_folder = "/models/OpenHermes-2.5-Mistral-7B"
image_output_folder = "/home/username/Desktop/"
```

This is self explanitory, but set the model folder to the location of your Mistral 7B, and the image output folder to the path you'd like to save your image.
