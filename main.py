import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import top_k_accuracy_score, accuracy_score

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=False, transform=transform)
test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_features = resnet50.fc.in_features
resnet50.fc = nn.Sequential(nn.Linear(num_features, 128), nn.Linear(128, 10)  )

optimizers = {
    'Adam',
    'Adagrad',
    'Adadelta',
    # 'RMSprop': optim.RMSprop(resnet50.parameters(), lr=0.001),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stored_params = resnet50.parameters()

resnet50 = resnet50.to(device)

def train_model(model, name, train_loader, test_loader, num_epochs=10):
    train_losses = []
    train_accs = []
    test_accs = []
    if name == "Adam":
        optimizer = optim.Adam(resnet50.parameters(), lr=0.001)
    elif name == "Adagrad":
        optimizer = optim.Adagrad(resnet50.parameters(), lr=0.001)
    else:
        optimizer = optim.Adadelta(resnet50.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')


    scores_list = []
    predictions = []
    all_labels = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = outputs.to(device)
            outputs = model(inputs)


            # total_test += labels.size(0)
            # correct_test += (predicted == labels).sum().item()
            scores_list.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    _, predicted = torch.max(torch.tensor(scores_list), 1)

    scores_list=np.array(scores_list)
    all_labels=np.array(all_labels)
    predictions = np.array(predicted)

    test_accuracy = accuracy_score(all_labels, predictions)

    top_5_acc = top_k_accuracy_score(all_labels, scores_list, k=5)

    print(f"Test Acc: {test_accuracy:.4f}")

    return train_losses, train_accs, test_accs, top_5_acc

results = {}
for optimizer_name, optimizer in optimizers.items():
    print(f'Training with {optimizer_name}')
    # resnet50.fc.reset_parameters()
    for param_cur, param_best in zip(resnet50.parameters(), stored_params):
        param_cur.data = param_best.data

    train_losses, train_accs, test_accs, top_5_acc = train_model(resnet50, optimizer, train_loader, test_loader,num_epochs=2)
    results[optimizer_name] = {'train_losses': train_losses, 'train_accs': train_accs, 'test_accs': test_accs,'top_5_acc': top_5_acc}

plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(range(1, len(result['train_losses']) + 1), result['train_losses'], label=f'{optimizer_name} Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(range(1, len(result['train_accs']) + 1), result['train_accs'], label=f'{optimizer_name} Train Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()

print(results)

for item in results:
    print("Top 5 Acc. =",  results[item]["top_5_acc"])