import os
import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Driver name to FastF1 code mapping
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

# Create cache folder
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# Get combined driver average data from multiple years
def get_multi_year_driver_averages(years, gp_name):
    all_laps = []
    for year in years:
        try:
            session = fastf1.get_session(year, gp_name, "R")
            session.load()
            laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
            laps.dropna(inplace=True)
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                laps[f"{col} (s)"] = laps[col].dt.total_seconds()
            all_laps.append(laps)
        except Exception as e:
            print(f"[{year} - {gp_name}] Skipped due to error: {e}")
    if all_laps:
        combined = pd.concat(all_laps)
        return combined.groupby("Driver")[
            ["LapTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
        ].mean().reset_index()
    else:
        return pd.DataFrame()

# Prediction logic for a race using historic data
def predict_race(location_name):
    print(f"\n📦 Loading historic data for: {location_name}")
    avg_times = get_multi_year_driver_averages([2022, 2023, 2024], location_name)

    if avg_times.empty:
        print(f"❌ No data found for {location_name} in given years.")
        return

    merged = QUALI_2025.merge(avg_times, left_on="DriverCode", right_on="Driver", how="left")
    merged = merged.rename(columns={"Driver_x": "Driver"})
    merged.dropna(subset=["LapTime (s)"] , inplace=True)

    X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
    y = merged["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    merged["PredictedRaceTime (s)"] = model.predict(X)
    results = merged[["Driver", "PredictedRaceTime (s)"]].sort_values(by="PredictedRaceTime (s)")

    print(f"\n🏁 Predicted 2025 {location_name} GP Results 🏁\n")
    print(results.to_string(index=False))

    y_pred = model.predict(X_test)
    print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

if __name__ == "__main__":
    races = [
        "Australia", "China", "Japan", "Bahrain",
        "Miami", "Emilia Romagna", "Monaco", "Spain", "Canada", "Austria",
        "United Kingdom", "Belgium", "Hungary", "Netherlands", "Italy",
        "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil",
        "Las Vegas", "Qatar", "Abu Dhabi"
    ]

    print("\n🏁 2025 Race List (excluding Saudi Arabia):")
    for i, race in enumerate(races, 1):
        print(f"{i:2}. {race}")

    selected = input("\nPlease enter the name of a Grand Prix to predict (e.g. 'Japan'): ").strip()
    if selected in races:
        predict_race(selected)
    else:
        print("❌ Invalid race name.")
