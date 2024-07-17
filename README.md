Prerequisites  
1. WS Game Company [Scrabble Set](https://www.wsgamecompany.com/scrabble-deluxe-edition.html)
2. Windows OS (only runs on Windows)  
3. iVCAM (installed both in camera device and host machine)
4. CUDA-capable GPU (without the GPU, training and executing will be inefficient)
5. CUDA Toolkit (>= v.12.5)
6. package dependencies listed in requirements.txt
      
Steps  
1. ensure package dependencies are installed
2. run train.py in kpt_rcnn_ml to generate the .pth file for the neural network for corner detection (finding the 4 corners of the Scrabble board) (it is not provided because the neural network is too large)
3. open iVCAM on the camera device and make sure the Scrabble board is in view  
4. run scrabble_assistant.bat to start the program   
