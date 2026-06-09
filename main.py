import torch
from data_loader import get_dataloaders, CLASSES
from train import train_model
from evaluate import evaluate_model
from utils import plot_training_history, plot_confusion_matrix, compare_results
from models.simple_cnn import SimpleCNN
from models.googlenet import GoogLeNet
from models.resnet import ResNet

# 설정
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용 장치: {DEVICE}")

# 데이터 로드
print("\n데이터 로딩 중...")
loaders = get_dataloaders(batch_size=BATCH_SIZE)

# 모델 학습 및 평가 함수
def run_experiment(model, model_name):
    print(f"\n{'='*50}")
    print(f"{model_name} 실험 시작")
    print(f"{'='*50}")

    results = {}

    for mode in ['no_norm', 'norm']:
        label = '정규화 없음' if mode == 'no_norm' else '정규화 있음'
        print(f"\n[{label}] 학습 시작...")

        # 매 실험마다 모델 초기화
        m = type(model)().to(DEVICE)

        # 학습
        trained_model, history = train_model(
            m, loaders[mode], num_epochs=NUM_EPOCHS, lr=LR, device=DEVICE
        )

        # 평가
        acc, cm = evaluate_model(trained_model, loaders[mode]['test'], device=DEVICE)
        print(f"[{label}] Test Accuracy: {acc:.2f}%")

        # 그래프 저장
        plot_training_history(history, model_name=f"{model_name}_{mode}",
                              save_path=f"{model_name}_{mode}_history.png")
        plot_confusion_matrix(cm, CLASSES, model_name=f"{model_name}_{mode}",
                              save_path=f"{model_name}_{mode}_cm.png")

        results[mode] = acc

    return results

# 실험 실행
all_results = {}

# SimpleCNN (베이스라인 확인용)
all_results['SimpleCNN'] = run_experiment(SimpleCNN(), 'SimpleCNN')

# GoogLeNet
all_results['GoogLeNet'] = run_experiment(GoogLeNet(), 'GoogLeNet')

# ResNet
all_results['ResNet'] = run_experiment(ResNet(), 'ResNet')

# 최종 결과 비교
print("\n\n최종 결과 요약")
print("="*50)
for model_name, res in all_results.items():
    print(f"{model_name} | 정규화 없음: {res['no_norm']:.2f}% | 정규화 있음: {res['norm']:.2f}%")

compare_results(all_results)