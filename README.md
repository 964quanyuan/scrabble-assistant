### DEMO: [click here](https://964quanyuan.github.io/scrabble-assistant/)

### Prerequisites  
1. WS Game Company [Scrabble Set](https://www.wsgamecompany.com/scrabble-deluxe-edition.html)
2. Windows OS (only runs on Windows)  
3. iVCAM (installed both in camera device and host machine)
4. CUDA-capable GPU (without the GPU, training and executing will be inefficient)
5. CUDA Toolkit (version >= 12.5)
6. package dependencies listed in requirements.txt
      
### Setup  
1. ensure package dependencies are installed
2. run train.py in kpt_rcnn_ml to generate the CNN (.pth file) for gameboard corner detection (not provided because the CNN is too large)  
3. open iVCAM on the device used to scan the gameboard and secure the device according to the diagram shown in the demo link  
4. make sure the Scrabble board is in view  
5. run scrabble_assistant.bat to start the program   
