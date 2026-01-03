"""
Question 3: Deep Neural Network Classifier for Handwritten Digits
This script implements:
- PyTorch DigitClassifier neural network
- Training loop with Adam optimizer
- Evaluation and visualization
- Theoretical explanations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Model Architecture
# ============================================================================

class DigitClassifier(nn.Module):
    """
    Deep Neural Network for digit classification
    
    Architecture:
    - Input: 784-dimensional vector (28x28 flattened image)
    - Hidden Layer 1: 256 neurons, ReLU activation
    - Hidden Layer 2: 128 neurons, ReLU activation
    - Output Layer: 10 logits (digits 0-9)
    """
    
    def __init__(self):
        super(DigitClassifier, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(784, 256)  # Input to hidden layer 1
        self.relu1 = nn.ReLU()           # ReLU activation
        
        self.fc2 = nn.Linear(256, 128)   # Hidden layer 1 to hidden layer 2
        self.relu2 = nn.ReLU()           # ReLU activation
        
        self.fc3 = nn.Linear(128, 10)    # Hidden layer 2 to output
        
        # Note: No softmax here - CrossEntropyLoss includes it
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, 784)
            Input batch of flattened images
            
        Returns:
        --------
        out : torch.Tensor of shape (batch_size, 10)
            Output logits for each class
        """
        # Flatten input if needed
        x = x.view(-1, 784)
        
        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        out = self.fc3(x)
        
        return out

# ============================================================================
# Training Loop
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001, device='cpu'):
    """
    Train the neural network
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for Adam optimizer
    device : str
        Device to train on ('cpu' or 'cuda')
        
    Returns:
    --------
    history : dict
        Training history with losses and accuracies
    """
    
    print("\n" + "="*70)
    print("TRAINING DEEP NEURAL NETWORK")
    print("="*70)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nConfiguration:")
    print(f"   Loss Function: CrossEntropyLoss")
    print(f"   Optimizer: Adam")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {train_loader.batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {device}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"\n{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            loss = criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Average training metrics
        avg_train_loss = train_loss / train_total
        train_accuracy = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                outputs = model(data)
                
                # Compute loss
                loss = criterion(outputs, target)
                
                # Statistics
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Average validation metrics
        avg_val_loss = val_loss / val_total
        val_accuracy = 100.0 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Print progress
        print(f"{epoch+1:<10} {avg_train_loss:<15.4f} {train_accuracy:<15.2f}% {avg_val_loss:<15.4f} {val_accuracy:<15.2f}%")
    
    print("\n✓ Training complete!")
    
    return history

# ============================================================================
# Evaluation and Visualization
# ============================================================================

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained model
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    device : str
        Device to evaluate on
        
    Returns:
    --------
    accuracy : float
        Test accuracy
    predictions : numpy array
        Predicted labels
    true_labels : numpy array
        True labels
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    model.eval()
    model = model.to(device)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confusion matrix to confusion_matrix.png")
    plt.close()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=[f'Digit {i}' for i in range(10)]))
    
    return accuracy, predictions, true_labels

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation loss/accuracy curves
    
    Parameters:
    -----------
    history : dict
        Training history
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training history to {save_path}")
    plt.close()

def visualize_predictions(model, test_loader, device='cpu', num_samples=16):
    """
    Visualize some predictions
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    device : str
        Device
    num_samples : int
        Number of samples to visualize
    """
    model.eval()
    model = model.to(device)
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.cpu().numpy()
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        img = images[idx].reshape(28, 28)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        
        color = 'green' if predictions[idx] == labels[idx] else 'red'
        axes[idx].set_title(f'True: {labels[idx]}, Pred: {predictions[idx]}', 
                           color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved predictions visualization to predictions_visualization.png")
    plt.close()

# ============================================================================
# Theoretical Questions
# ============================================================================

def print_theoretical_answers():
    """
    Print answers to theoretical questions
    """
    print("\n" + "="*70)
    print("THEORETICAL QUESTIONS")
    print("="*70)
    
    print("\n1. Why is ReLU preferred over Sigmoid and Tanh in deep networks?")
    print("   Provide two reasons.")
    print()
    print("   ANSWER:")
    print()
    print("   Reason 1: MITIGATING THE VANISHING GRADIENT PROBLEM")
    print("   -------------------------------------------------------")
    print("   • ReLU has a gradient of 1 for all positive inputs:")
    print("     ReLU(x) = max(0, x)")
    print("     d/dx ReLU(x) = 1 if x > 0, else 0")
    print()
    print("   • Sigmoid and Tanh have gradients that saturate:")
    print("     σ(x) = 1/(1 + e^(-x))")
    print("     d/dx σ(x) = σ(x)(1 - σ(x)) → 0 as |x| → ∞")
    print()
    print("     tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))")
    print("     d/dx tanh(x) = 1 - tanh²(x) → 0 as |x| → ∞")
    print()
    print("   • In deep networks with many layers, gradients are multiplied")
    print("     during backpropagation. With sigmoid/tanh, these small gradients")
    print("     (< 0.25) get multiplied many times, causing vanishing gradients.")
    print()
    print("   • ReLU maintains gradient flow through deep networks because")
    print("     its gradient doesn't diminish for positive activations.")
    print()
    print("   Reason 2: COMPUTATIONAL EFFICIENCY")
    print("   -----------------------------------")
    print("   • ReLU is computationally cheaper:")
    print("     - Forward pass: Simple thresholding operation (max(0, x))")
    print("     - Backward pass: Binary mask (gradient is 0 or 1)")
    print()
    print("   • Sigmoid/Tanh require expensive exponential computations:")
    print("     - Sigmoid: e^(-x) calculation")
    print("     - Tanh: Multiple exponentials (e^x and e^(-x))")
    print()
    print("   • In large neural networks with millions of neurons,")
    print("     this computational difference becomes significant.")
    print()
    print("   Additional benefits of ReLU:")
    print("   • Sparse activation (some neurons output 0)")
    print("   • More biologically plausible")
    print("   • Faster convergence in practice")
    print()
    
    print("\n2. Explain the role of PyTorch's autograd engine, including how it")
    print("   builds computation graphs and performs backpropagation.")
    print()
    print("   ANSWER:")
    print()
    print("   PYTORCH'S AUTOGRAD ENGINE")
    print("   ==========================")
    print()
    print("   A. COMPUTATION GRAPH CONSTRUCTION (Dynamic Graph)")
    print("   --------------------------------------------------")
    print()
    print("   1. Define-by-Run Paradigm:")
    print("      • PyTorch uses dynamic computation graphs (unlike TensorFlow 1.x)")
    print("      • Graph is built on-the-fly during forward pass")
    print("      • Each operation creates a new node in the graph")
    print()
    print("   2. Tensor with requires_grad=True:")
    print("      • Marks tensors for automatic differentiation")
    print("      • PyTorch tracks all operations on these tensors")
    print()
    print("      Example:")
    print("      x = torch.tensor([1.0, 2.0], requires_grad=True)")
    print("      y = x * 2        # Creates computation node: MulBackward")
    print("      z = y.mean()     # Creates computation node: MeanBackward")
    print()
    print("   3. Function Objects:")
    print("      • Each operation stores a 'grad_fn' (gradient function)")
    print("      • Links form a Directed Acyclic Graph (DAG)")
    print("      • Stores necessary information for backward pass")
    print()
    print("   B. COMPUTATION GRAPH STRUCTURE")
    print("   --------------------------------")
    print()
    print("   The graph stores:")
    print("   • Inputs to each operation (for gradient computation)")
    print("   • The operation type (add, multiply, ReLU, etc.)")
    print("   • Links to parent nodes")
    print()
    print("   Example graph for z = (x * w + b).relu():")
    print()
    print("      x (requires_grad=True)")
    print("       ↓")
    print("      MulBackward0  ← stores x, w")
    print("       ↓")
    print("      AddBackward0  ← stores intermediate, b")
    print("       ↓")
    print("      ReluBackward0 ← stores mask of positive values")
    print("       ↓")
    print("      z (result)")
    print()
    print("   C. BACKPROPAGATION ALGORITHM")
    print("   -----------------------------")
    print()
    print("   When you call loss.backward():")
    print()
    print("   1. Initialize gradient:")
    print("      • Start at the loss (scalar)")
    print("      • Initial gradient: dL/dL = 1")
    print()
    print("   2. Traverse graph in reverse (topological order):")
    print("      • Start from loss, work backwards to inputs")
    print("      • Visit each node in reverse order of creation")
    print()
    print("   3. Apply chain rule at each node:")
    print("      • For node computing y = f(x):")
    print("        dL/dx = dL/dy × dy/dx")
    print()
    print("      • Each grad_fn knows how to compute local gradient dy/dx")
    print("      • Multiply by incoming gradient dL/dy")
    print()
    print("   4. Accumulate gradients:")
    print("      • Store computed gradient in tensor.grad")
    print("      • If tensor is used multiple times, gradients are summed")
    print()
    print("   D. EXAMPLE WALKTHROUGH")
    print("   -----------------------")
    print()
    print("   Forward pass:")
    print("   x = torch.tensor([2.0], requires_grad=True)")
    print("   w = torch.tensor([3.0], requires_grad=True)")
    print("   b = torch.tensor([1.0], requires_grad=True)")
    print()
    print("   y = x * w      # y = 6.0")
    print("   z = y + b      # z = 7.0")
    print("   loss = z ** 2  # loss = 49.0")
    print()
    print("   Backward pass (loss.backward()):")
    print()
    print("   Step 1: dL/dL = 1")
    print()
    print("   Step 2: PowerBackward")
    print("      dL/dz = dL/dL × d(z²)/dz = 1 × 2z = 2 × 7 = 14")
    print()
    print("   Step 3: AddBackward")
    print("      dL/dy = dL/dz × dz/dy = 14 × 1 = 14")
    print("      dL/db = dL/dz × dz/db = 14 × 1 = 14")
    print()
    print("   Step 4: MulBackward")
    print("      dL/dx = dL/dy × dy/dx = 14 × w = 14 × 3 = 42")
    print("      dL/dw = dL/dy × dy/dw = 14 × x = 14 × 2 = 28")
    print()
    print("   Final gradients:")
    print("   x.grad = 42")
    print("   w.grad = 28")
    print("   b.grad = 14")
    print()
    print("   E. KEY FEATURES")
    print("   ----------------")
    print()
    print("   1. Automatic Differentiation:")
    print("      • No manual gradient implementation needed")
    print("      • Reduces errors and development time")
    print()
    print("   2. Dynamic Graphs:")
    print("      • Different graph each forward pass")
    print("      • Enables conditional logic, loops")
    print("      • Easier debugging (Python's debugger works)")
    print()
    print("   3. Gradient Accumulation:")
    print("      • Gradients accumulate in .grad")
    print("      • Must call optimizer.zero_grad() before each step")
    print()
    print("   4. Memory Efficiency:")
    print("      • Intermediate values released when no longer needed")
    print("      • Can use torch.no_grad() to disable tracking")
    print()
    print("   5. Higher-Order Gradients:")
    print("      • Can compute gradients of gradients")
    print("      • Useful for second-order optimization")
    print()
    print("   F. PRACTICAL IMPLICATIONS")
    print("   --------------------------")
    print()
    print("   In the training loop:")
    print()
    print("   # Forward pass - builds computation graph")
    print("   outputs = model(inputs)")
    print("   loss = criterion(outputs, targets)")
    print()
    print("   # Backward pass - computes gradients")
    print("   loss.backward()")
    print()
    print("   # Optimizer updates parameters using computed gradients")
    print("   optimizer.step()")
    print()
    print("   # Clear gradients for next iteration")
    print("   optimizer.zero_grad()")
    print()
    print("   This elegant API abstracts away the complex mathematics of")
    print("   backpropagation while maintaining computational efficiency.")
    print()

# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Main function to execute Question 3
    """
    print("\n" + "="*70)
    print("QUESTION 3: DEEP NEURAL NETWORK CLASSIFIER")
    print("="*70)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load or generate data
    try:
        from torchvision import datasets, transforms
        
        print("\n✓ Loading MNIST dataset...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, 
                                     download=True, transform=transform)
        
        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"\n⚠ Could not load MNIST: {e}")
        print("  Generating synthetic data for demonstration...")
        
        # Generate synthetic data
        n_train = 5000
        n_val = 1000
        n_test = 1000
        
        X_train = torch.randn(n_train, 784) * 0.5
        y_train = torch.randint(0, 10, (n_train,))
        
        X_val = torch.randn(n_val, 784) * 0.5
        y_val = torch.randint(0, 10, (n_val,))
        
        X_test = torch.randn(n_test, 784) * 0.5
        y_test = torch.randint(0, 10, (n_test,))
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        print(f"   Generated synthetic data:")
        print(f"   Training samples: {n_train}")
        print(f"   Validation samples: {n_val}")
        print(f"   Test samples: {n_test}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    print("\n✓ Initializing DigitClassifier model...")
    model = DigitClassifier()
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_model(model, train_loader, val_loader, 
                         epochs=5, learning_rate=0.001, device=device)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    accuracy, predictions, true_labels = evaluate_model(model, test_loader, device=device)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'digit_classifier.pth')
    print("\n✓ Saved model to digit_classifier.pth")
    
    # Print theoretical answers
    print_theoretical_answers()
    
    print("\n" + "="*70)
    print("QUESTION 3 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - predictions_visualization.png")
    print("  - digit_classifier.pth (model weights)")
    print("\nAll theoretical answers are printed above and included in the LaTeX report.")

if __name__ == "__main__":
    main()
