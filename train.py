from src.data_cleaning import load_and_clean_data
from src.model import train_model

df = load_and_clean_data("data/hotel_bookings.csv")
artifacts = train_model(df)

print("Model trained successfully!")
print(f"MAE: {artifacts['metrics']['mae']}")
print(f"RMSE: {artifacts['metrics']['rmse']}")
print(f"R2: {artifacts['metrics']['r2']}")
