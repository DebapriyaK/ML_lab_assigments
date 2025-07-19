# stock_analysis.py

import statistics
import matplotlib.pyplot as plt
from data_processing import load_irctc_data

def main():
    df = load_irctc_data("Lab Session Data.xlsx")

    #1: Mean and Variance of entire Price column 
    price_data = df["Price"].dropna()
    population_mean = statistics.mean(price_data)
    population_variance = statistics.variance(price_data)
    print(f"Population Mean (Price): {population_mean:.2f}")
    print(f"Population Variance (Price): {population_variance:.2f}")

    # 2 Mean of Price for all Wednesdays 
    wednesday_prices = df[df["Weekday_Abbr"] == "Wed"]["Price"].dropna()
    wednesday_mean = statistics.mean(wednesday_prices)
    print(f"\nSample Mean (Wednesdays): {wednesday_mean:.2f}")
    print("Observation:", "Higher than" if wednesday_mean > population_mean else "Lower than", "population mean.")

    # 3 Mean of Price for April month 
    april_prices = df[df["Month"] == "April"]["Price"].dropna()
    if len(april_prices) > 0:
        april_mean = statistics.mean(april_prices)
        print(f"\nSample Mean (April): {april_mean:.2f}")
        print("Observation:", "Higher than" if april_mean > population_mean else "Lower than", "population mean.")
    else:
        print("\nNo April data found in dataset.")

    # 4 Probability of loss (Chg% < 0) 
    chg = df["Chg%"].dropna()
    prob_loss = sum(chg < 0) / len(chg)
    print(f"\nProbability of loss: {prob_loss:.2f}")

    # 5Probability of profit on Wednesday 
    wed_data = df[df["Weekday_Abbr"] == "Wed"]["Chg%"].dropna()
    prob_profit_wed = sum(wed_data > 0) / len(wed_data)
    print(f"Probability of profit on Wednesday: {prob_profit_wed:.2f}")

    # 6 Conditional probability of profit given it's Wednesday
    prob_profit_given_wed = prob_profit_wed  # Same as Task 5
    print(f"Conditional Probability (Profit | Wednesday): {prob_profit_given_wed:.2f}")

    # 7Scatter Plot of Chg% vs Weekday 
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Weekday_Abbr"], df["Chg%"], color='blue', alpha=0.6)
    plt.xlabel("Day of the Week")
    plt.ylabel("Change (%)")
    plt.title("Chg% vs Day of the Week")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
