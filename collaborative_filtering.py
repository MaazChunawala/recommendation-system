import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens dataset
# Adjust the path if needed
df = pd.read_csv("/Users/alichunawala/Downloads/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# Prepare the data for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use Singular Value Decomposition (SVD) for collaborative filtering
model = SVD()

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
print("RMSE: ", accuracy.rmse(predictions))

# Example prediction for a specific user and item
user_id = 1
item_id = 50
predicted_rating = model.predict(user_id, item_id).est
print(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating}")