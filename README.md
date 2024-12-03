This is fast implementation of small LLM following Andrej Karpathy's instructions form: https://www.youtube.com/watch?v=kCc8FmEb1nY

Structure:  
main.py includes the structure of the LLM.  
train.py can be run to train a model yourself, from given text file.  
test.py can be used to get few paragraph from trained module with random start input.  

If you want to test the code with model that I have already trained you can find it in: https://drive.google.com/file/d/1LVY3OsjbE-j8vxgfKe7bJdnas7AE2NWq/view?usp=sharing

You should drop it into new folder "model_folder" or change the model_path variable from test.py file.  
The model has been trained with a relatively long light novel that I happen to like called "Lord of the Mysteries".  
The model produces mostly legible text and clearly demonstrate that it atleast knows few of the main characters by name. Personally I call it a success for the training recourses I have available. 
