import os
import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Map full driver names to FastF1 driver codes
DRIVER_MAPPING = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

# Hypothetical 2025 Qualifying Data
QUALI_2025 = pd.DataFrame({
    "Driver": list(DRIVER_MAPPING.keys()),
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                            91.021, 91.079, 91.103, 91.638, 91.706,
                            91.625, 91.632, 91.688, 91.773, 91.840,
                            91.992, 92.018, 92.092, 92.141, 92.174]
})
QUALI_2025["DriverCode"] = QUALI_2025["Driver"].map(DRIVER_MAPPING)

# Prediction Function
def predict_race(location_name):
    print(f"\n📦 Loading data for: {location_name} 2024")

    os.makedirs("f1_cache", exist_ok=True)
    fastf1.Cache.enable_cache("f1_cache")

    try:
        session = fastf1.get_session(2024, location_name, "R")
        session.load()
    except Exception as e:
        print(f"❌ Error loading {location_name}: {e}")
        return

    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(inplace=True)

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()

    avg_times = laps.groupby("Driver")[
        ["LapTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    ].mean().reset_index()

    merged = QUALI_2025.merge(avg_times, left_on="DriverCode", right_on="Driver", how="left")
    merged = merged.rename(columns={"Driver_x": "Driver"})
    merged.dropna(subset=["LapTime (s)"], inplace=True)

    X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
    y = merged["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
    model.fit(X_train, y_train)

    merged["PredictedRaceTime (s)"] = model.predict(X)
    results = merged[["Driver", "PredictedRaceTime (s)"]].sort_values(by="PredictedRaceTime (s)")

    print(f"\n🏁 Predicted 2025 {location_name} GP Winner 🏁\n")
    print(results.to_string(index=False))

    y_pred = model.predict(X_test)
    print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Example usage with user input
if __name__ == "__main__":
    races = [
        "Australia", "China", "Japan", "Bahrain",
        "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada", "Austria",
        "United Kingdom", "Belgium", "Hungary", "Netherlands", "Italy",
        "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil",
        "Las Vegas", "Qatar", "Abu Dhabi"
    ]
    print("\n🏁 2025 Season Race List (excluding Saudi Arabia):")
    for i, r in enumerate(races, start=1):
        print(f"{i:2}. {r}")

    choice = input("\nPlease enter a Grand Prix name (e.g. 'Japan'): ").strip()
    if choice in races:
        predict_race(choice)
    else:
        print("❌ Invalid race name.")

