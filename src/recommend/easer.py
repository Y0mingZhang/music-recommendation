import json, argparse

import numpy as np
import torch

from recommend.utils import *

data_directory = "data/filtered"


class EaseR:
    def __init__(self, X_train: np.ndarray, lambda_: float):
        X_train = self.transform_interaction_matrix(X_train)
        self.B = self.optimize(X_train, lambda_)

    @timer
    def optimize(self, X: np.ndarray, lambda_: float) -> np.ndarray:
        G = X.T.dot(X)
        U, I = X.shape
        diagIndices = np.diag_indices(I)
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        return B

    def transform_interaction_matrix(self, X: np.ndarray) -> np.ndarray:
        return (X >= 75).astype(float)

    def recommend(self, X_test: np.ndarray, k: int = 100) -> np.ndarray:
        X_test = self.transform_interaction_matrix(X_test)
        scores = X_test @ self.B
        topk = np.argsort(-scores, axis=1)[:, :k]
        return topk


class EaseRGradientDescent:
    def __init__(
        self,
        X_train: np.ndarray,
        lambda_: float,
        lr: float,
        num_epochs: int,
        batch_size: int,
    ):
        X_train = self.transform_interaction_matrix(X_train)
        self.B = self.optimize(X_train, lambda_, lr, num_epochs, batch_size)

    @timer
    def optimize(
        self, X: np.ndarray, lambda_: float, lr: float, num_epochs: int, batch_size: int
    ) -> np.ndarray:
        assert torch.cuda.is_available()

        U, I = X.shape
        B = torch.zeros((I, I), requires_grad=True, device="cuda")
        X = torch.FloatTensor(X).cuda()
        X_dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(
            X_dataset, batch_size=batch_size, shuffle=True
        )
        opt = torch.optim.SGD([B], lr=lr)

        for epochs in range(num_epochs):
            for x_batch in dataloader:
                x = x_batch[0]
                x = x.to("cuda")
                opt.zero_grad()
                recon_loss = torch.norm(torch.matmul(x, B) - x) ** 2 / batch_size * U
                reg_loss = lambda_ * torch.norm(B) ** 2
                loss = recon_loss + reg_loss
                loss.backward()
                B.grad.fill_diagonal_(0)
                opt.step()

        return B.detach().cpu().numpy()

    def transform_interaction_matrix(self, X: np.ndarray) -> np.ndarray:
        return (X >= 75).astype(float)

    def recommend(self, X_test: np.ndarray, k: int = 100) -> np.ndarray:
        X_test = self.transform_interaction_matrix(X_test)
        scores = X_test @ self.B
        topk = np.argsort(-scores, axis=1)[:, :k]
        return topk


def main():
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/filtered", type=str)
    parser.add_argument("--alg", choices=["easer", "easer-gd"], default="easer")
    parser.add_argument("--lambda_", default=100.0, type=float)
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()

    print("config")
    print(json.dumps(args.__dict__))

    album_df, user_dfs = load_dataframes(args.data_dir)
    (user_train_df, user_val_df, user_test_df) = user_dfs

    X_train = generate_user_item_matrix(user_train_df, album_df)
    X_val = generate_user_item_matrix(user_val_df, album_df)
    X_test = generate_user_item_matrix(user_test_df, album_df)

    if args.alg == "easer":
        rec_alg = EaseR(X_train, args.lambda_)
    else:
        rec_alg = EaseRGradientDescent(
            X_train, args.lambda_, args.lr, args.num_epochs, args.batch_size
        )

    rec_result = rec_alg.recommend(X_val)
    summary = evaluate(rec_result, X_val)
    print("val performance")
    print(json.dumps(summary, indent=2))

    rec_result = rec_alg.recommend(X_test)
    summary = evaluate(rec_result, X_test)
    print("test performance")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
