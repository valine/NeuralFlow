# Neural Flow

This is a python script to plot the intermediate layer output of Mistral 7B. When you run the script it will produce a 512x256 image representing the output at every layer of the model. 

The idea here is simple: collect the output tensors at every layer, normalize between zero and one, and plot those values as a heat map. The resulting image has a surprising amount of structure, and I have found it enormously helpful when fine-tuning models as a way to visually inspect the output.

Here is the visualization from mistral 7B out of the box with no fine tuning.

![probe_results_layers_20240212_221145](https://github.com/valine/NeuralFlow/assets/14074844/ab939cc2-a5fa-4a1a-8e45-bc2b5741f0e1)

Intentionally over fitting a model on a small fine-tuning dataset yields the following output. You can see a problem start around layer 10 which cascade through the remaining layers.

![overfit_output](https://github.com/valine/NeuralFlow/assets/14074844/c6788265-5c8c-45ba-8092-98ec6d3caf09)

The true value of this visualization are the patterns that become apparent when comparing the output before and after training. By periodically visualizing the model output it's possible to create an animation of the models intermidate output over time. 

https://github.com/valine/NeuralFlow/assets/14074844/1f2e50ea-d64d-4f37-a991-f968399e29bd



# Constants
There are two file paths you will want to change before running the script:

```
model_folder = "/models/OpenHermes-2.5-Mistral-7B"
image_output_folder = "/home/username/Desktop/"
```

This is self explanitory, but set the model folder to the location of your Mistral 7B, and the image output folder to the path you'd like to save your image.
