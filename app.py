from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load the models
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))

features = ['DriverID', 'TeamID', 'GP_ID', 'Year', 'Round','QualPosition', 'Driver_AvgFinish_Season',
             'Team_AvgPoints_Season','PrevRaceFinish', 'Form_Last3','AvgFinish_AtGP_PastYears', 
             'AvgQual_AtGP_PastYears','Drivers_Confidence']

app = Flask(__name__)

# Predicts top 3 drivers (Winner, Second, Third) for a given race dataframe.
def predict_podium(model, race_df):
    
    X_race = race_df[features]
    probs = model.predict_proba(X_race)

    # Convert probabilities into dataframe
    prob_df = pd.DataFrame(probs, columns=model.classes_, index=race_df.index)

    # We want probability of finishing at position 1
    if 1 in prob_df.columns:
        race_df['Win_Prob'] = prob_df[1]
    else:
        race_df['Win_Prob'] = 0

    # Rank drivers by win probability
    top3 = race_df.sort_values('Win_Prob', ascending=False).head(3)

    # Return as list of tuples (Driver, Team, Probability)
    return list(top3[['Driver', 'TeamName', 'Win_Prob']].itertuples(index=False, name=None))


@app.route('/',methods=['GET', 'POST'])
def index():
    rf_pred, xgb_pred, error = None, None, None

    if request.method == 'POST':
        gp_name = request.form['GrandPrix']

        try:
            # Load last known 2024 GP data for selected GP
            df = pd.read_csv("Cleaned_F1_Data.csv")  # your cleaned dataset
            print(df['GrandPrix'].str.strip().unique())  # Clean whitespace
            gp_history = df[df['GrandPrix'] == gp_name]

            if gp_history.empty:
                error = f"No data found for {gp_name}"
            else:
                last_year = gp_history['Year'].max()
                race_data = gp_history[gp_history['Year'] == last_year].copy()

                # Simulate 2025 GP using last year's drivers/teams
                race_data['Year'] = 2025
                race_data['Round'] = race_data['Round'].max() + 1

                # Predict podium
                rf_pred = predict_podium(rf_model, race_data)
                xgb_pred = predict_podium(xgb_model, race_data)

        except Exception as e:
            error = f"Error processing GP '{gp_name}': {e}"

    return render_template("form.html", rf=rf_pred, xgb=xgb_pred, error=error)


if __name__ == '__main__':
    app.run(debug=True)