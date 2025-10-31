import torch

SEPT_ACCURACY = 98.79

def train_model(model, training_data, epochs, optimizer, loss_fuc, device):
    for epoch in range(epochs):
        model.train()

        for batch_i, (data, target) in enumerate(training_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = loss_fuc(output, target)
            loss.backward()
            optimizer.step()
            
        if batch_i % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_i * len(data)}/{len(training_data.dataset)} ({100. * batch_i / len(training_data.dataset):.0f}%)]\tLoss: {loss.item():.6f}")


def test_model(model, testing_data, loss_fuc, device):
    model.eval()

    test_loss = correct = 0

    with torch.no_grad():
        for data, target in testing_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fuc(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testing_data)

    # accuracy measurement
    accuracy = (100. * correct) / (len(testing_data.dataset))
    print(f"\nTest Set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(testing_data.dataset)} ({100. * correct / len(testing_data.dataset):.2f}%\n)")

    return accuracy
