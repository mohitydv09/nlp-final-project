## nlp-final-project
Repo for the final project of NLP 

# World Coordinate extraction


* IMage here *




## VLM Setup:

BLIP Image Captioning:

* Image here *

**Unconditional caption:** there is a man standing in a room with a whiteboard
**Conditional caption:** a photography of a man standing in a room with a whiteboard

Project Proposal Report Feedback:

- Good job on the thorough experiment description.

Thank You

- The title and focus may also be misleading. How are you going to utilize your approach in a way specific to visual impairment? If the end result of your work does not directly solve this issue, then I would suggest re-structuring your paper to instead focus more on just getting better general scene descriptions.

We want to focus on this specific task i.e. scene description methods already exits, we want to focus on there relevance for a visual impaired person. 

Example: There might be a tree in the distance and there might be a trash can on the street close by while a general scene descriptor will focus on both the objects and might not even inform the user about the trash can. We aim to focus on objects that will be relevant for a visually impaired person to know about in their day-to-day life. As in the example of the trash can, it might be important for the user not to collide with it rather than the tree in the distance.

- The only thing missing here is a clearer description of the metrics you will be using to evaluate the models on their downstream tasks. Given the visual impairment objective, is there an existing dataset you can use to see how well your model performs on this task specifically. If not, can you construct one and (in some way) come up with a meaningful evaluation pipeline?). Another point of comparison is that you should compare your model results with VLMs that have come out more recently.

We could not find a dataset specifically pertaining to our task. As of now, we plan to conduct a human evaluation on our model output and compare it with general VLMs. We aim to perform a blind test (user won’t be informed about the method used to generate the output) with human subjects to judge the compare output of our model with a VLM in various scenarios.

In addition to the text generation comparison, we will evaluate the processing time using our model versus the baseline VLM model. The process time becomes much more important when it pertains to providing important alerts to the visually impaired user. 

Current Step:

1. Setting Up LLMs to Produce a coherent output based on the objects and their locations in the scene.
    
    example:
    
    A dog and a human are currently standing close by, with the dog slightly to the left and the human to the right. A van is a bit further away on the right side. It seems like the dog is moving towards you, while the van is moving away.
    

Next Steps:

1. Evaluation Pipeline to evaluate the model with a VLM.
