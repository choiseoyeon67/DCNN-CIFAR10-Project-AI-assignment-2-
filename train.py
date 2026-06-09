import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloaders, num_epochs=20, lr=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_acc  = 100. * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        log_message = (f"Epoch [{epoch+1}/{num_epochs}] "
                       f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        if 'val' in dataloaders:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            epoch_val_loss = val_loss / len(dataloaders['val'])
            epoch_val_acc = 100. * val_correct / val_total
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)

            log_message += (f" | Val Loss: {epoch_val_loss:.4f} "
                            f"| Val Acc: {epoch_val_acc:.2f}%")

        print(log_message)

    return model, history
