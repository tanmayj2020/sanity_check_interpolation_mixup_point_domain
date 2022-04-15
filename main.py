import torch
from model import Sketch_Classification
from dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sketch Classification")

    parser.add_argument(
        "--base_dir", type=str, default=os.getcwd(), help="In order to access from condor"
    )
    parser.add_argument(
        "--saved_models", type=str, default="./models", help="Saved models directory"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="TUBerlin", help="TUBerlin vs Sketchy"
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="Resnet",
        help="VGG / InceptionV3/ Resnet",
    )
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--nThreads", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--eval_freq_iter", type=int, default=1)
    parser.add_argument("--print_freq_iter", type=int, default=1)
    parser.add_argument("--splitTrain", type=float, default=0.7)
    parser.add_argument(
        "--training", type=str, default="sketch", help="sketch / rgb / edge"
    )
    parser.add_argument(
        "--channels", type=int, default=3, help="channel of input image"
    )

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)

    model = Sketch_Classification(hp)
    print(model.Network)
    model.to(device)
    step = 0
    best_accuracy = 0

    # os.makedirs(hp.saved_models, exist_ok=True)

    for epoch in range(hp.max_epoch):

        for i_batch, batch in enumerate(dataloader_Train):
            loss = model.train_model(batch)
            print(f"Epoch - {epoch} , Batch - {i_batch} , Loss - {loss}")

        if epoch % hp.print_freq_iter == 0:
            print(
                "Epoch: {}, Loss: {}, Best Accuracy: {}".format(
                    epoch, loss, best_accuracy
                )
            )

        if epoch % hp.eval_freq_iter == 0:
            with torch.no_grad():
                accuracy = model.evaluate(dataloader_Test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        hp.saved_models, "model_best_" + str(hp.training) + ".pth"
                    ),
                )
    print("Best Accuracy: {}".format(best_accuracy))

        
