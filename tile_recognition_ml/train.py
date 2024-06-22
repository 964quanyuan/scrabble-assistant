import torch
from torch.autograd import Variable
from network import Network, loss_fn, classes
from torch.optim import Adam
from torch.utils.data import DataLoader
from data_loader import ImageFolderDataset

# Function to save the model
def saveModel():
    path = "./wonderful_explosions.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.unsqueeze(1)
            labels = labels.squeeze(1)
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            correct_predictions = (predicted == labels.to(device))
            accuracy += correct_predictions.sum().item()

            # print the labels that fail to match
            failed_labels = labels[~correct_predictions]
            predicted_failed = predicted[~correct_predictions]
            for actual, predicted in zip(failed_labels, predicted_failed):
                print(f"Actual: {classes[actual.item()]}, Predicted: {classes[predicted.item()]}")
            
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # 3d -> 4d
            images = images.unsqueeze(1)
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            #print(outputs.shape, labels.shape)
            labels = labels.squeeze(1)
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %.2f %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

if __name__ == "__main__":
    train_set = ImageFolderDataset(root_dir="dataset", train=True)
    test_set = ImageFolderDataset(root_dir="dataset", train=False)
    #sample_image = train_set.__getitem__(133)
    #print(sample_image[1].shape)  # Assuming channels are at index 0

    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=1)
    print("The number of images in a training set is: ", len(train_loader)*8)

    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=1)
    print("The number of images in a test set is: ", len(test_loader)*8)

    print("The number of batches per epoch is: ", len(train_loader))

    # Instantiate a neural network model 
    model = Network()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train(20)
    print('Finished Training')
    
    # testAccuracy()