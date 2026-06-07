import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, model_name='Model', save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 그래프
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy 그래프
    ax2.plot(epochs, history['train_acc'], 'r-o', label='Train Acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, classes, model_name='Model', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    # 각 칸에 숫자 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compare_results(results):
    """
    results: dict
    예시) {'SimpleCNN': {'no_norm': 72.3, 'norm': 75.1}, ...}
    """
    models = list(results.keys())
    no_norm_accs = [results[m]['no_norm'] for m in models]
    norm_accs    = [results[m]['norm']    for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, no_norm_accs, width, label='No Normalization')
    ax.bar(x + width/2, norm_accs,    width, label='With Normalization')

    ax.set_title('Model Accuracy Comparison (Normalization)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()