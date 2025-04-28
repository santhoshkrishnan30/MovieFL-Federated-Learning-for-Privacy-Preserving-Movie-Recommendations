import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
import time
import requests
import io
import sys
import os
# Temporarily comment out Syft to avoid initialization error
# import syft as sy

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

def add_noise(model, noise_scale=0.01):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)
    return model

def local_train(model, train_data, epochs=5, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        losses = []
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
            losses.append(loss.item())
        print(f"Epoch {epoch+1} - Loss: {np.mean(losses):.4f}")

    return add_noise(model)

def start_client(client_id, server_addr="localhost", server_port=5000, max_retries=3):
    # Register with server using HTTPS
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://{server_addr}/register_client",
                json={'client_id': client_id},
                timeout=10,
                verify=False
            )
            if response.status_code != 200:
                print(f"[{client_id}] Registration failed: Status {response.status_code}, Response: {response.text}")
                return
            data = response.json()
            print(f"[{client_id}] Successfully registered with server")
            n_users = data.get("n_users", 940)  # Default to a reasonable value
            n_movies = data.get("n_movies", 1411)  # Default to a reasonable value
            break
        except requests.exceptions.RequestException as e:
            print(f"[{client_id}] Registration attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print(f"[{client_id}] Failed to register after {max_retries} attempts. Exiting.")
                return
            time.sleep(2)

    # Load data and init model with server-provided sizes
    train_df, _, _, _ = load_data(sample_frac=0.05)
    model = RecSysModel(n_users, n_movies)
    print(f"[{client_id}] Ready for training with n_users={n_users}, n_movies={n_movies}")
    last_round = -1

    while True:
        try:
            # Check server status using HTTPS
            response = requests.get(
                f"https://{server_addr}/metrics",
                timeout=10,
                verify=False
            )
            status = response.json()
            print(f"[{client_id}] Server status: current_round={status['current_round']}, training_active={status['training_active']}, ready_models={status['ready_models']}")
            
            if status.get('current_round', 0) >= status.get('max_rounds', 5):
                print(f"[{client_id}] Training completed")
                break
                
            # Train and submit if a new round is detected
            if status['current_round'] > last_round:
                print(f"[{client_id}] Starting round {status['current_round'] + 1}")
                model = local_train(model, train_df)
                
                # Submit model directly using torch.save
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            f"https://{server_addr}/submit_model",
                            files={'model_file': buffer},
                            data={'client_id': client_id},
                            timeout=10,
                            verify=False
                        )
                        buffer.seek(0)  # Reset buffer for retry if needed
                        if response.status_code == 200:
                            print(f"[{client_id}] Model submitted successfully")
                            last_round = status['current_round']
                            break
                        else:
                            print(f"[{client_id}] Upload failed: Status {response.status_code}, Response: {response.text}")
                    except requests.exceptions.RequestException as e:
                        print(f"[{client_id}] Submission attempt {attempt+1}/{max_retries} failed: {e}")
                        if attempt == max_retries - 1:
                            print(f"[{client_id}] Failed to submit model after {max_retries} attempts.")
                            break
                        time.sleep(2)
            
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"[{client_id}] Network error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[{client_id}] Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True, help='Unique client ID')
    parser.add_argument('--server-ip', default="localhost", help='Server IP')
    parser.add_argument('--server-port', type=int, default=5000, help='Server port')
    
    args = parser.parse_args()
    start_client(args.client_id, args.server_ip, args.server_port)