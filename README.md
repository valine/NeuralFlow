# Neural Flow

This is a python script to plot the intermediate layer output of Mistral 7B. When you run the script it will produce a 512x256 image representing the output at every layer of the model. 

The idea here is simple: collect the output tensors at every layer, normalize between zero and one, and plot those values as a heat map. The resulting image has a surprising amount of structure, and I have found it enormously helpful when fine-tuning models as a way to visually inspect the output. 


### Visualizing the model during training

The following is a visualization from Mistral 7B, out of the box with no fine tuning.

![initial_output](https://github.com/valine/NeuralFlow/assets/14074844/aef6a0fc-820c-4e6d-94df-a907df8a7018)


Intentionally over fitting a model on a small fine-tuning dataset yields the following output. You can see a problem start around layer 10 which cascade through the remaining layers.

![overfit_output](https://github.com/valine/NeuralFlow/assets/14074844/c6788265-5c8c-45ba-8092-98ec6d3caf09)

The true value of this visualization are the patterns that become apparent when comparing the output before and after training. By periodically visualizing the model output it's possible to create an animation of the models intermidate output over time. Failures within a single layer cascade to the high layers. While it's difficult ascribe meaning to the structures within the visualization, it's visually apparent when the distribution of output has deviated from the initial state.

https://github.com/valine/NeuralFlow/assets/14074844/1f2e50ea-d64d-4f37-a991-f968399e29bd

### How the image is structured

The resolution and structure of this visualization warrents additional explanation. The intermediate output of Mistral 7B for a single token is a 4096 dimention tensor for each of the 32 layers. For the purposes of visualizaition a 4096x32 image is impractical. To solve this, I have chopped up the image into chucks of 512, and arranged them vertically. The end result is a 512x256 image that displays nicely on landscape displays.

![guide](https://github.com/valine/NeuralFlow/assets/14074844/7cf5ad4a-98a7-4ec4-896c-fe4fb5068654)


# Constants
There are two file paths you will want to change before running the script:

```
model_folder = "/models/OpenHermes-2.5-Mistral-7B"
image_output_folder = "/home/username/Desktop/"
```

This is self explanitory, but set the model folder to the location of your Mistral 7B, and the image output folder to the path you'd like to save your image.
