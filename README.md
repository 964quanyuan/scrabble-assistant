Prerequisites  
1. WS Game Company [Scrabble Set] (https://www.wsgamecompany.com/scrabble-deluxe-edition.html)
2. Windows OS (only runs on Windows)  
3. iVCAM (installed both in camera device and host machine)
4. CUDA-capable GPU (without the GPU, training and executing will take far too long)
5. CUDA Toolkit (>= v.12.5)
6. pytorch
      
Steps  
1. run train.py in kpt_rcnn_ml to generate the .pth file for the neural network for corner detection (finding the 4 corners of the Scrabble board) (it is not provided because the file is too large)
2. open iVCAM on the camera device and make sure the Scrabble board is in view  
3. run scrabble_assistant.bat to start the program   
