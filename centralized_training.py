import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.fc = nn.Linear(n_factors * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, movie_ids):
        user_vecs = self.user_embedding(user_ids)
        movie_vecs = self.movie_embedding(movie_ids)
        x = torch.cat([user_vecs, movie_vecs], dim=1)
        x = self.fc(x)
        return self.sigmoid(x) * 5

def load_data(sample_frac=0.2, random_state=42):
    ratings = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", 
                         sep="\t", header=None,
                         names=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings.sample(frac=sample_frac, random_state=random_state)

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    ratings["userId"] = user_encoder.fit_transform(ratings["userId"])
    ratings["movieId"] = movie_encoder.fit_transform(ratings["movieId"])

    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()

    train, test = train_test_split(ratings, test_size=0.05, random_state=random_state)
    return train, test, n_users, n_movies

def train_centralized(model, train_data, test_data, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for idx in range(0, len(train_data), 32):
            batch = train_data.iloc[idx:idx+32]
            user = torch.LongTensor(batch["userId"].values)
            movie = torch.LongTensor(batch["movieId"].values)
            rating = torch.FloatTensor(batch["rating"].values).view(-1,1)

            optimizer.zero_grad()
            pred = model(user, movie)
            loss = loss_fn(pred, rating)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            user = torch.LongTensor(test_data["userId"].values)
            movie = torch.LongTensor(test_data["movieId"].values)
            rating = torch.FloatTensor(test_data["rating"].values).view(-1,1)
            pred = model(user, movie)
            mse = loss_fn(pred, rating)
            rmse = torch.sqrt(mse).item()
            acc = (1 - rmse/5)
            accuracies.append(acc)

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} | RMSE={rmse:.4f} | Accuracy={acc:.4f}")

    torch.save({'losses': losses, 'accuracies': accuracies}, 'centralized_metrics.pt')
    torch.save(model.state_dict(), 'centralized_model.pt')
    
    return losses, accuracies

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    
    train_data, test_data, n_users, n_movies = load_data()
    model = RecSysModel(n_users, n_movies)
    losses, accuracies = train_centralized(model, train_data, test_data)

    epochs = list(range(1, len(losses)+1))
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(epochs, losses, marker='o')
    plt.title('Centralized Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracies, marker='o', color='green')
    plt.title('Centralized Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('static/centralized_training_result.png')
    plt.show()